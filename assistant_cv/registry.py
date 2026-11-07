"""Persistent Person Registry for BMO-CV Long-Term Identity Management."""

from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Any
import numpy as np

try:
    from .config import GALLERY_DIR, OBJECT_CLASSES_PATH, OBJECT_GALLERY_DIR, OBJECT_TAU
except ImportError:
    from config import GALLERY_DIR, OBJECT_CLASSES_PATH, OBJECT_GALLERY_DIR, OBJECT_TAU


def _sanitize_id(name: str) -> str:
    """Convert name to filesystem and registry-safe identifier."""
    clean = re.sub(r"[^\w\s-]", "", name.strip()).strip().lower()
    clean = re.sub(r"[-\s]+", "_", clean)
    return clean or "user"


class IdentityRegistry:
    """Manages persistent identity profiles in gallery/registry.json alongside embeddings.
    
    Identities registered here are NEVER automatically deleted or pruned.
    They remain permanent until an explicit manual delete command is issued.
    """

    def __init__(self, gallery_dir: pathlib.Path | str = GALLERY_DIR) -> None:
        self.gallery_dir = pathlib.Path(gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.gallery_dir / "registry.json"
        self.identities: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> dict[str, dict[str, Any]]:
        """Load registry from JSON file and synchronize with *.npy files on disk."""
        data: dict[str, Any] = {}
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"[IdentityRegistry] Warning loading registry: {exc}")

        loaded_identities = data.get("identities", {})

        # Synchronize with any existing *.npy files in gallery_dir
        for npy_path in sorted(self.gallery_dir.glob("*.npy")):
            file_id = _sanitize_id(npy_path.stem)
            if file_id not in loaded_identities:
                # Add discovered identity
                loaded_identities[file_id] = {
                    "id": file_id,
                    "name": npy_path.stem,
                    "created_at": npy_path.stat().st_mtime,
                    "last_seen": npy_path.stat().st_mtime,
                    "encounters": 1,
                    "file": npy_path.name,
                }
            else:
                loaded_identities[file_id]["file"] = npy_path.name

        self.identities = loaded_identities
        self.save()
        return self.identities

    def save(self) -> None:
        """Atomically persist registry data to disk."""
        tmp_file = self.registry_file.with_suffix(".tmp")
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "identities": self.identities,
        }
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_file.replace(self.registry_file)
        except Exception as exc:
            print(f"[IdentityRegistry] Error saving registry: {exc}")
            if tmp_file.exists():
                tmp_file.unlink()

    def is_registered(self, name_or_id: str) -> bool:
        """Check if an identity exists in the registry."""
        norm_key = _sanitize_id(name_or_id)
        if norm_key in self.identities:
            return True
        for ident in self.identities.values():
            if str(ident.get("name", "")).strip().lower() == name_or_id.strip().lower():
                return True
        return False

    def get_identity(self, name_or_id: str) -> dict[str, Any] | None:
        """Get profile for a registered identity."""
        norm_key = _sanitize_id(name_or_id)
        if norm_key in self.identities:
            return self.identities[norm_key]
        for ident in self.identities.values():
            if str(ident.get("name", "")).strip().lower() == name_or_id.strip().lower():
                return ident
        return None

    def register_person(
        self,
        name: str,
        embedding: np.ndarray,
        overwrite: bool = True,
        notes: str = "",
    ) -> tuple[bool, str, pathlib.Path]:
        """Save a person's embedding and record them in the persistent registry."""
        clean_name = name.strip()
        if not clean_name:
            clean_name = f"Person_{len(self.identities) + 1}"

        ident_id = _sanitize_id(clean_name)
        now = time.time()

        if ident_id in self.identities and not overwrite:
            return False, f"Identity '{clean_name}' is already registered.", self.gallery_dir / self.identities[ident_id]["file"]

        # Normalize embedding vector
        e = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm_val = float(np.linalg.norm(e))
        if norm_val > 1e-8:
            e = e / norm_val

        out_path = self.gallery_dir / f"{clean_name}.npy"
        np.save(out_path, e)

        existing = self.identities.get(ident_id)
        created_at = existing["created_at"] if existing else now
        encounters = (existing["encounters"] + 1) if existing else 1
        entry_notes = notes if notes else (existing.get("notes", "") if existing else "")

        self.identities[ident_id] = {
            "id": ident_id,
            "name": clean_name,
            "created_at": created_at,
            "last_seen": now,
            "encounters": encounters,
            "notes": entry_notes,
            "file": out_path.name,
        }
        self.save()
        return True, f"Successfully registered '{clean_name}'.", out_path

    get_person = get_identity

    def delete_person(self, name_or_id: str) -> tuple[bool, str]:
        """Manually delete a registered person. Only called on explicit user command."""
        ident = self.get_identity(name_or_id)
        if ident is None:
            return False, f"Person '{name_or_id}' not found in registry."

        ident_id = ident["id"]
        file_name = ident.get("file", f"{ident.get('name', ident_id)}.npy")
        npy_path = self.gallery_dir / file_name

        if npy_path.exists():
            try:
                npy_path.unlink()
            except Exception as exc:
                print(f"[IdentityRegistry] Error removing {npy_path}: {exc}")

        del self.identities[ident_id]
        self.save()
        return True, f"Permanently deleted person '{ident.get('name', ident_id)}'."

    def rename_person(self, old_name_or_id: str, new_name: str) -> tuple[bool, str]:
        """Rename an existing registered identity."""
        ident = self.get_identity(old_name_or_id)
        if ident is None:
            return False, f"Person '{old_name_or_id}' not found."

        new_clean = new_name.strip()
        if not new_clean:
            return False, "New name cannot be empty."

        old_id = ident["id"]
        new_id = _sanitize_id(new_clean)

        old_file = self.gallery_dir / ident.get("file", f"{ident['name']}.npy")
        new_file = self.gallery_dir / f"{new_clean}.npy"

        if old_file.exists():
            try:
                old_file.rename(new_file)
            except Exception as exc:
                print(f"[IdentityRegistry] Error renaming file: {exc}")

        ident["name"] = new_clean
        ident["id"] = new_id
        ident["file"] = new_file.name

        if new_id != old_id:
            del self.identities[old_id]
            self.identities[new_id] = ident

        self.save()
        return True, f"Renamed '{old_name_or_id}' to '{new_clean}'."

    def record_encounter(self, name_or_id: str) -> None:
        """Update encounter count and last_seen timestamp."""
        ident = self.get_identity(name_or_id)
        if ident is not None:
            ident["last_seen"] = time.time()
            ident["encounters"] = ident.get("encounters", 0) + 1
            # Debounce file save on frequent updates
            if ident["encounters"] % 10 == 0:
                self.save()

    def list_identities(self) -> list[dict[str, Any]]:
        """Return list of all registered identities."""
        return list(self.identities.values())

    def get_known_names(self) -> list[str]:
        """Return list of human-readable names of all registered persons."""
        return [ident.get("name", k) for k, ident in self.identities.items()]


_REGISTRY: IdentityRegistry | None = None


def get_identity_registry() -> IdentityRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = IdentityRegistry()
    return _REGISTRY


# ---------------------------------------------------------------------------
# Persistent Object Registry & Vocabulary Synchronization
# ---------------------------------------------------------------------------
class ObjectRegistry:
    """Manages persistent object profiles, synonyms, and visual exemplars in gallery/objects/."""

    def __init__(
        self,
        gallery_dir: pathlib.Path | str = OBJECT_GALLERY_DIR,
        classes_path: pathlib.Path | str = OBJECT_CLASSES_PATH,
    ) -> None:
        self.gallery_dir = pathlib.Path(gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.classes_path = pathlib.Path(classes_path)
        self.registry_file = self.gallery_dir / "registry.json"
        self.objects: dict[str, dict[str, Any]] = {}
        self._gallery_names: list[str] = []
        self._gallery_matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.load()

    def load(self) -> dict[str, dict[str, Any]]:
        """Load object profiles from registry.json and sync with visual *.npy files."""
        data: dict[str, Any] = {}
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"[ObjectRegistry] Warning loading registry: {exc}")

        loaded = data.get("objects", {})

        # Synchronize with any existing *.npy visual exemplars
        for npy_path in sorted(self.gallery_dir.glob("*.npy")):
            obj_id = _sanitize_id(npy_path.stem)
            if obj_id not in loaded:
                loaded[obj_id] = {
                    "id": obj_id,
                    "name": npy_path.stem,
                    "synonyms": [],
                    "created_at": npy_path.stat().st_mtime,
                    "last_seen": npy_path.stat().st_mtime,
                    "encounters": 0,
                    "has_visual_exemplar": True,
                    "file": npy_path.name,
                }
            else:
                loaded[obj_id]["has_visual_exemplar"] = True
                loaded[obj_id]["file"] = npy_path.name

        self.objects = loaded
        self.reload_gallery()
        return self.objects

    def reload_gallery(self) -> int:
        """Reload and normalize 512-D visual embeddings matrix from disk."""
        names: list[str] = []
        vecs: list[np.ndarray] = []

        for obj_id, info in self.objects.items():
            file_name = info.get("file")
            if not file_name:
                continue
            npy_path = self.gallery_dir / file_name
            if npy_path.exists():
                try:
                    vec = np.load(npy_path).astype(np.float32).reshape(-1)
                    norm_val = float(np.linalg.norm(vec))
                    if norm_val > 1e-8:
                        vec = vec / norm_val
                    vecs.append(vec)
                    names.append(info.get("name", obj_id))
                except Exception as exc:
                    print(f"[ObjectRegistry] Failed to load {npy_path}: {exc}")

        self._gallery_names = names
        if vecs:
            self._gallery_matrix = np.stack(vecs, axis=0).astype(np.float32)
        else:
            self._gallery_matrix = np.empty((0, 512), dtype=np.float32)

        return len(names)

    def save(self) -> None:
        """Atomically persist registry to registry.json and sync classes to everyday_objects.json."""
        payload = {"version": 1, "objects": self.objects}
        tmp_file = self.registry_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_file.replace(self.registry_file)
        except Exception as exc:
            print(f"[ObjectRegistry] Error saving registry: {exc}")
            if tmp_file.exists():
                tmp_file.unlink()

        # Synchronize active vocabulary to exterior everyday_objects.json
        self._sync_classes_file()

    def _sync_classes_file(self) -> None:
        """Ensure all registered object names and synonyms are present in everyday_objects.json."""
        if not self.classes_path.exists():
            current_classes = []
        else:
            try:
                with open(self.classes_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                if isinstance(cdata, list):
                    current_classes = [str(c).strip() for c in cdata if str(c).strip()]
                elif isinstance(cdata, dict) and "classes" in cdata:
                    current_classes = [str(c).strip() for c in cdata["classes"] if str(c).strip()]
                else:
                    current_classes = []
            except Exception:
                current_classes = []

        existing_lower = {c.lower() for c in current_classes}
        updated = False

        for info in self.objects.values():
            name = str(info.get("name", "")).strip()
            if name and name.lower() not in existing_lower:
                current_classes.append(name)
                existing_lower.add(name.lower())
                updated = True
            for syn in info.get("synonyms", []):
                syn_str = str(syn).strip()
                if syn_str and syn_str.lower() not in existing_lower:
                    current_classes.append(syn_str)
                    existing_lower.add(syn_str.lower())
                    updated = True

        if updated:
            try:
                tmp_c = self.classes_path.with_suffix(".tmp")
                with open(tmp_c, "w", encoding="utf-8") as f:
                    json.dump({"classes": current_classes}, f, indent=2, ensure_ascii=False)
                tmp_c.replace(self.classes_path)
            except Exception as exc:
                print(f"[ObjectRegistry] Error updating {self.classes_path}: {exc}")

    def is_registered(self, name_or_id: str) -> bool:
        """Check if an object is already registered by ID, name, or synonym."""
        norm_key = _sanitize_id(name_or_id)
        if norm_key in self.objects:
            return True
        for obj in self.objects.values():
            if str(obj.get("name", "")).strip().lower() == name_or_id.strip().lower():
                return True
            for syn in obj.get("synonyms", []):
                if str(syn).strip().lower() == name_or_id.strip().lower():
                    return True
        return False

    def get_object(self, name_or_id: str) -> dict[str, Any] | None:
        """Get profile for a registered object."""
        norm_key = _sanitize_id(name_or_id)
        if norm_key in self.objects:
            return self.objects[norm_key]
        for obj in self.objects.values():
            if str(obj.get("name", "")).strip().lower() == name_or_id.strip().lower():
                return obj
            for syn in obj.get("synonyms", []):
                if str(syn).strip().lower() == name_or_id.strip().lower():
                    return obj
        return None

    def register_object(
        self,
        name: str,
        synonyms: list[str] | None = None,
        embedding: np.ndarray | None = None,
        overwrite: bool = True,
    ) -> tuple[bool, str, pathlib.Path | None]:
        """Register a new object, save optional visual embedding, and sync vocabulary."""
        clean_name = name.strip()
        if not clean_name:
            clean_name = f"Object_{len(self.objects) + 1}"

        obj_id = _sanitize_id(clean_name)
        now = time.time()

        if obj_id in self.objects and not overwrite:
            return False, f"Object '{clean_name}' is already registered.", None

        out_path: pathlib.Path | None = None
        has_vis = False
        if embedding is not None:
            e = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm_val = float(np.linalg.norm(e))
            if norm_val > 1e-8:
                e = e / norm_val
            out_path = self.gallery_dir / f"{clean_name}.npy"
            np.save(out_path, e)
            has_vis = True

        existing = self.objects.get(obj_id)
        created_at = existing["created_at"] if existing else now
        encounters = (existing["encounters"] + 1) if existing else 1
        syns = list(synonyms) if synonyms is not None else (existing.get("synonyms", []) if existing else [])

        self.objects[obj_id] = {
            "id": obj_id,
            "name": clean_name,
            "synonyms": syns,
            "created_at": created_at,
            "last_seen": now,
            "encounters": encounters,
            "has_visual_exemplar": has_vis or (existing.get("has_visual_exemplar", False) if existing else False),
            "file": out_path.name if out_path else (existing.get("file") if existing else None),
        }

        self.save()
        self.reload_gallery()
        return True, f"Successfully registered object '{clean_name}'.", out_path

    def delete_object(self, name_or_id: str) -> tuple[bool, str]:
        """Manually delete a registered object and its visual exemplars."""
        obj = self.get_object(name_or_id)
        if obj is None:
            return False, f"Object '{name_or_id}' not found in registry."

        obj_id = obj["id"]
        file_name = obj.get("file")
        if file_name:
            npy_path = self.gallery_dir / file_name
            if npy_path.exists():
                try:
                    npy_path.unlink()
                except Exception as exc:
                    print(f"[ObjectRegistry] Error removing {npy_path}: {exc}")

        del self.objects[obj_id]
        self.save()
        self.reload_gallery()
        return True, f"Deleted object '{obj.get('name', obj_id)}'."

    def rename_object(self, old_name_or_id: str, new_name: str) -> tuple[bool, str]:
        """Rename an existing registered object."""
        obj = self.get_object(old_name_or_id)
        if obj is None:
            return False, f"Object '{old_name_or_id}' not found in registry."

        new_clean = new_name.strip()
        if not new_clean:
            return False, "New object name cannot be empty."

        old_id = obj["id"]
        new_id = _sanitize_id(new_clean)

        file_name = obj.get("file")
        if file_name:
            old_file = self.gallery_dir / file_name
            new_file = self.gallery_dir / f"{new_clean}.npy"
            if old_file.exists():
                try:
                    old_file.rename(new_file)
                    obj["file"] = new_file.name
                except Exception as exc:
                    print(f"[ObjectRegistry] Error renaming file: {exc}")

        obj["name"] = new_clean
        obj["id"] = new_id

        if new_id != old_id:
            del self.objects[old_id]
            self.objects[new_id] = obj

        self.save()
        self.reload_gallery()
        return True, f"Renamed '{old_name_or_id}' to '{new_clean}'."

    def check_already_registered(
        self,
        name: str | None = None,
        embedding: np.ndarray | None = None,
        tau: float = OBJECT_TAU,
    ) -> tuple[str | None, float]:
        """Check if an object matches an already registered name or visual exemplar."""
        if name and self.is_registered(name):
            obj = self.get_object(name)
            return (obj["name"] if obj else name), 1.0

        if embedding is not None and self._gallery_names and self._gallery_matrix.shape[0] > 0:
            emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm_val = float(np.linalg.norm(emb))
            if norm_val > 1e-8:
                emb = emb / norm_val
            sims = self._gallery_matrix @ emb
            best_idx = int(np.argmax(sims))
            score = float(sims[best_idx])
            if score >= tau:
                return self._gallery_names[best_idx], round(score, 3)

        return None, 0.0

    def match_visual(
        self,
        embedding: np.ndarray,
        tau: float = OBJECT_TAU,
    ) -> tuple[str | None, float]:
        """Match a crop visual embedding against registered exemplars."""
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
            return self._gallery_names[best_idx], round(score, 3)
        return None, round(score, 3)

    def record_encounter(self, name_or_id: str) -> None:
        """Update encounter count and last_seen timestamp."""
        obj = self.get_object(name_or_id)
        if obj is not None:
            obj["last_seen"] = time.time()
            obj["encounters"] = obj.get("encounters", 0) + 1
            if obj["encounters"] % 10 == 0:
                self.save()

    def list_objects(self) -> list[dict[str, Any]]:
        """Return list of all registered object profiles."""
        return list(self.objects.values())

    def get_known_names(self) -> list[str]:
        """Return list of canonical names of all registered objects."""
        return [obj.get("name", k) for k, obj in self.objects.items()]


_OBJECT_REGISTRY: ObjectRegistry | None = None


def get_object_registry() -> ObjectRegistry:
    global _OBJECT_REGISTRY
    if _OBJECT_REGISTRY is None:
        _OBJECT_REGISTRY = ObjectRegistry()
    return _OBJECT_REGISTRY
