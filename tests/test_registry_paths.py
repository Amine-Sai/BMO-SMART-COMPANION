"""Tests for path traversal blocking and safe gallery writes in registry.py."""

from __future__ import annotations

import pathlib
import numpy as np
import pytest

from assistant_cv.registry import IdentityRegistry, ObjectRegistry, _sanitize_id


def test_sanitize_id() -> None:
    assert _sanitize_id("John Doe") == "John_Doe"
    assert _sanitize_id("  Judas  ") == "Judas"
    assert _sanitize_id("../evil") == "evil"
    assert _sanitize_id("/abs") == "abs"
    assert _sanitize_id("   ") == "user"
    assert _sanitize_id("special@item#1!") == "specialitem1"


@pytest.fixture
def identity_gallery(tmp_path: pathlib.Path) -> pathlib.Path:
    gal = tmp_path / "gallery"
    gal.mkdir(parents=True, exist_ok=True)
    return gal


@pytest.fixture
def object_gallery(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    gal = tmp_path / "object_gallery"
    gal.mkdir(parents=True, exist_ok=True)
    classes_file = tmp_path / "classes.json"
    return gal, classes_file


@pytest.fixture
def dummy_embedding() -> np.ndarray:
    vec = np.random.randn(512).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


def test_identity_register_blocks_path_traversal(identity_gallery: pathlib.Path, dummy_embedding: np.ndarray) -> None:
    reg = IdentityRegistry(gallery_dir=identity_gallery)

    traversal_names = ["../evil", "/abs", "../../evil", "evil/name", "..", "/root_inject"]
    for evil_name in traversal_names:
        with pytest.raises(ValueError, match="Path traversal detected"):
            reg.register_person(evil_name, dummy_embedding)

    # Verify no file escaped gallery_dir
    parent_evil = identity_gallery.parent / "evil.npy"
    assert not parent_evil.exists()
    root_abs = pathlib.Path("/abs.npy")
    assert not root_abs.exists()


def test_object_register_blocks_path_traversal(
    object_gallery: tuple[pathlib.Path, pathlib.Path],
    dummy_embedding: np.ndarray,
) -> None:
    gal, classes_path = object_gallery
    reg = ObjectRegistry(gallery_dir=gal, classes_path=classes_path)

    traversal_names = ["../evil", "/abs", "../../evil", "evil/item", ".."]
    for evil_name in traversal_names:
        # With embedding
        with pytest.raises(ValueError, match="Path traversal detected"):
            reg.register_object(evil_name, embedding=dummy_embedding)

        # Without embedding
        with pytest.raises(ValueError, match="Path traversal detected"):
            reg.register_object(evil_name, embedding=None)

    parent_evil = gal.parent / "evil.npy"
    assert not parent_evil.exists()


def test_rename_blocks_path_traversal(
    identity_gallery: pathlib.Path,
    object_gallery: tuple[pathlib.Path, pathlib.Path],
    dummy_embedding: np.ndarray,
) -> None:
    ident_reg = IdentityRegistry(gallery_dir=identity_gallery)
    ident_reg.register_person("Alice", dummy_embedding)

    with pytest.raises(ValueError, match="Path traversal detected"):
        ident_reg.rename_person("Alice", "../evil")
    with pytest.raises(ValueError, match="Path traversal detected"):
        ident_reg.rename_person("Alice", "/abs")
    with pytest.raises(ValueError, match="Path traversal detected"):
        ident_reg.rename_person("../evil", "NewAlice")

    gal, classes_path = object_gallery
    obj_reg = ObjectRegistry(gallery_dir=gal, classes_path=classes_path)
    obj_reg.register_object("mug", embedding=dummy_embedding)

    with pytest.raises(ValueError, match="Path traversal detected"):
        obj_reg.rename_object("mug", "../evil")
    with pytest.raises(ValueError, match="Path traversal detected"):
        obj_reg.rename_object("mug", "/abs")
    with pytest.raises(ValueError, match="Path traversal detected"):
        obj_reg.rename_object("../evil", "new_mug")


def test_delete_blocks_path_traversal(
    identity_gallery: pathlib.Path,
    object_gallery: tuple[pathlib.Path, pathlib.Path],
) -> None:
    ident_reg = IdentityRegistry(gallery_dir=identity_gallery)
    with pytest.raises(ValueError, match="Path traversal detected"):
        ident_reg.delete_person("../evil")
    with pytest.raises(ValueError, match="Path traversal detected"):
        ident_reg.delete_person("/abs")

    gal, classes_path = object_gallery
    obj_reg = ObjectRegistry(gallery_dir=gal, classes_path=classes_path)
    with pytest.raises(ValueError, match="Path traversal detected"):
        obj_reg.delete_object("../evil")
    with pytest.raises(ValueError, match="Path traversal detected"):
        obj_reg.delete_object("/abs")


def test_sanitized_id_in_file_paths(
    identity_gallery: pathlib.Path,
    object_gallery: tuple[pathlib.Path, pathlib.Path],
    dummy_embedding: np.ndarray,
) -> None:
    ident_reg = IdentityRegistry(gallery_dir=identity_gallery)
    ok, msg, path = ident_reg.register_person("John Doe", dummy_embedding)
    assert ok
    assert path.resolve().is_relative_to(identity_gallery.resolve())
    assert path.name == "John_Doe.npy"
    assert path.exists()

    gal, classes_path = object_gallery
    obj_reg = ObjectRegistry(gallery_dir=gal, classes_path=classes_path)
    ok_obj, msg_obj, path_obj = obj_reg.register_object("coffee mug", embedding=dummy_embedding)
    assert ok_obj
    assert path_obj is not None
    assert path_obj.resolve().is_relative_to(gal.resolve())
    assert path_obj.name == "coffee_mug.npy"
    assert path_obj.exists()


def test_loader_keeps_reading_existing_raw_named_files(
    identity_gallery: pathlib.Path,
    dummy_embedding: np.ndarray,
) -> None:
    # Simulate pre-existing gallery file named by raw name
    legacy_file = identity_gallery / "Legacy Raw Name.npy"
    np.save(legacy_file, dummy_embedding)

    reloaded = IdentityRegistry(gallery_dir=identity_gallery)
    assert reloaded.is_registered("Legacy Raw Name")
    entry = reloaded.get_identity("Legacy Raw Name")
    assert entry is not None
    assert entry["file"] == "Legacy Raw Name.npy"
