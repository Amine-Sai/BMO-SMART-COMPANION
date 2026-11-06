"""Face detection and recognition using InsightFace (buffalo_s) ONNX runtime."""

from __future__ import annotations

import sys
import time
from pathlib import Path

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

_FACE_APP: FaceAnalysis | None = None
_IDENTIFIER: FaceIdentifier | None = None


def get_face_app() -> FaceAnalysis:
    """Lazily initialize and return the InsightFace buffalo_s FaceAnalysis instance."""
    global _FACE_APP
    if _FACE_APP is None:
        app = FaceAnalysis(name="buffalo_s", providers=get_onnx_providers())
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
