"""Face detection and recognition using InsightFace (buffalo_s) ONNX runtime."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from assistant_cv import hud
from assistant_cv.config import (
    CAM_INDEX,
    DEVICE,
    FACE_TAU,
    GALLERY_DIR,
    get_onnx_providers,
)
from assistant_cv.registry import IdentityRegistry

_FACE_APP: FaceAnalysis | None = None
_IDENTIFIER: FaceIdentifier | None = None


def get_face_app() -> FaceAnalysis:
    """Lazily initialize and return the InsightFace buffalo_s FaceAnalysis instance."""
    global _FACE_APP
    if _FACE_APP is None:
        app = FaceAnalysis(
            name="buffalo_s",
            allowed_modules=["detection", "recognition"],
            providers=get_onnx_providers(),
        )
        app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
        _FACE_APP = app
    return _FACE_APP


class FaceIdentifier:
    """Matches detected faces against normalized 512-D gallery embeddings."""

    def __init__(
        self,
        gallery_dir: Path = GALLERY_DIR,
        app: FaceAnalysis | None = None,
    ) -> None:
        self.gallery_dir = Path(gallery_dir)
        self.registry = IdentityRegistry(gallery_dir=self.gallery_dir)
        self.app: FaceAnalysis = app if app is not None else get_face_app()
        self.gallery: dict[str, np.ndarray] = {}
        self._gallery_names: list[str] = []
        self._gallery_matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.reload_gallery()

    def reload_gallery(self) -> dict[str, np.ndarray]:
        """Reload all ``*.npy`` embeddings from ``self.gallery_dir``."""
        gallery: dict[str, np.ndarray] = {}
        names: list[str] = []
        vecs: list[np.ndarray] = []

        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.registry.load()

        for npy_path in sorted(self.gallery_dir.glob("*.npy")):
            vec = np.load(npy_path).astype(np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            gallery[npy_path.stem] = vec
            names.append(npy_path.stem)
            vecs.append(vec)

        self.gallery = gallery
        self._gallery_names = names
        if vecs:
            self._gallery_matrix = np.stack(vecs, axis=0).astype(np.float32)
        else:
            self._gallery_matrix = np.empty((0, 512), dtype=np.float32)
        return self.gallery

    def check_already_registered(
        self,
        embedding: np.ndarray,
        tau: float = FACE_TAU,
    ) -> tuple[str | None, float]:
        """Check if an embedding matches an already registered identity."""
        if not self._gallery_names or self._gallery_matrix.shape[0] == 0:
            return None, 0.0

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm_val = float(np.linalg.norm(emb))
        if norm_val > 1e-8:
            emb = emb / norm_val

        sims = self._gallery_matrix @ emb
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])
        if score >= tau:
            matched_name = self._gallery_names[best_idx]
            return matched_name, round(score, 3)
        return None, round(score, 3)

    def identify(
        self,
        frame: np.ndarray,
        tau: float = FACE_TAU,
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Detect and identify faces in ``frame``.

        Returns a list of ``(name, score, (x1, y1, x2, y2))`` tuples.
        """
        if frame is None:
            return []

        faces = self.app.get(frame)
        results: list[tuple[str, float, tuple[int, int, int, int]]] = []

        for f in faces:
            name = "unknown"
            score = 0.0

            if self._gallery_names and self._gallery_matrix.shape[0] > 0:
                emb = np.asarray(f.normed_embedding, dtype=np.float32).reshape(-1)
                sims = self._gallery_matrix @ emb
                best_idx = int(np.argmax(sims))
                score = float(sims[best_idx])
                name = self._gallery_names[best_idx]

            bbox_coords = tuple(f.bbox.astype(int).tolist())
            bbox: tuple[int, int, int, int] = (
                int(bbox_coords[0]),
                int(bbox_coords[1]),
                int(bbox_coords[2]),
                int(bbox_coords[3]),
            )
            results.append(
                (
                    name if score >= tau else "unknown",
                    round(float(score), 3),
                    bbox,
                )
            )

        return results

    def identify_and_embed(
        self,
        frame: np.ndarray,
        tau: float = FACE_TAU,
    ) -> list[dict[str, Any]]:
        """Detect and identify faces, returning identities with their embeddings."""
        if frame is None:
            return []

        faces = self.app.get(frame)
        results: list[dict[str, Any]] = []

        for f in faces:
            name = "unknown"
            score = 0.0
            emb = np.asarray(f.normed_embedding, dtype=np.float32).reshape(-1)
            norm_val = float(np.linalg.norm(emb))
            if norm_val > 1e-8:
                emb = emb / norm_val

            if self._gallery_names and self._gallery_matrix.shape[0] > 0:
                sims = self._gallery_matrix @ emb
                best_idx = int(np.argmax(sims))
                score = float(sims[best_idx])
                name = self._gallery_names[best_idx]

            bbox_coords = tuple(f.bbox.astype(int).tolist())
            bbox: tuple[int, int, int, int] = (
                int(bbox_coords[0]),
                int(bbox_coords[1]),
                int(bbox_coords[2]),
                int(bbox_coords[3]),
            )
            results.append(
                {
                    "name": name if score >= tau else "unknown",
                    "score": round(float(score), 3),
                    "bbox": bbox,
                    "embedding": emb,
                }
            )

        return results

    def save_identity(
        self,
        name: str,
        embeddings: list[np.ndarray],
        overwrite: bool = True,
    ) -> Path:
        """Average multiple embeddings, register in persistent registry, and reload gallery."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Identity name must not be empty.")
        if not embeddings:
            raise ValueError("Cannot save identity with zero embeddings.")

        e = np.mean(embeddings, axis=0).astype(np.float32)
        norm_val = float(np.linalg.norm(e))
        if norm_val > 1e-8:
            e = e / norm_val

        ok, msg, out_path = self.registry.register_person(clean_name, e, overwrite=overwrite)
        if not ok and not overwrite:
            raise ValueError(msg)

        self.reload_gallery()
        return out_path

    def delete_identity(self, name_or_id: str) -> tuple[bool, str]:
        """Manually delete a registered identity permanently."""
        ok, msg = self.registry.delete_person(name_or_id)
        if ok:
            self.reload_gallery()
        return ok, msg

    def rename_identity(self, old_name: str, new_name: str) -> tuple[bool, str]:
        """Rename an existing registered identity."""
        ok, msg = self.registry.rename_person(old_name, new_name)
        if ok:
            self.reload_gallery()
        return ok, msg


def _get_identifier() -> FaceIdentifier:
    global _IDENTIFIER
    if _IDENTIFIER is None:
        _IDENTIFIER = FaceIdentifier()
    return _IDENTIFIER


def identify(
    frame: np.ndarray,
    tau: float = FACE_TAU,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Module-level convenience wrapper around the singleton ``FaceIdentifier``."""
    return _get_identifier().identify(frame, tau=tau)


def reload_gallery() -> dict[str, np.ndarray]:
    """Reload gallery embeddings on the singleton ``FaceIdentifier``."""
    return _get_identifier().reload_gallery()


def main() -> None:
    identifier = _get_identifier()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[faces] Error: could not open camera index {CAM_INDEX}", file=sys.stderr)
        return

    fps = 0.0
    prev_t = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t0 = time.perf_counter()
            results = identifier.identify(frame)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            inst_fps = 1.0 / dt
            fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

            hud.draw_faces(frame, results)
            hud.draw_status_bar(
                frame,
                fps,
                device=DEVICE,
                toggles={"R:RELOAD": True},
                latencies={"face": infer_ms},
            )

            cv2.imshow("BMO-CV — Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                identifier.reload_gallery()
                print(f"[faces] Reloaded gallery ({len(identifier.gallery)} identities)")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
