"""Owner-scoped durable blobs. References contain no filesystem paths or embedded data."""

import json
import os
import re
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    owner: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size: int = Field(ge=0)
    expires_at: float | None = None


class ArtifactStore:
    def __init__(self, root, *, max_size=32 * 1024 * 1024):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_size = max_size

    def put(
        self, data: bytes, *, owner: str, media_type: str, ttl: float | None = None
    ) -> ArtifactRef:
        if len(data) > self.max_size:
            raise ValueError("Artifact exceeds size limit")
        if not re.fullmatch(r"[\w.+-]+/[\w.+-]+", media_type):
            raise ValueError("Invalid media type")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")
        ref = ArtifactRef(
            id=uuid.uuid4().hex,
            owner=owner,
            media_type=media_type,
            size=len(data),
            expires_at=time.time() + ttl if ttl else None,
        )
        blob = self.root / (ref.id + ".blob")
        with blob.open("xb") as f:
            os.chmod(blob, 0o600)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        meta = self.root / (ref.id + ".json")
        temp = self.root / (ref.id + ".tmp")
        with temp.open("x", encoding="utf-8") as f:
            os.chmod(temp, 0o600)
            f.write(ref.model_dump_json())
            f.flush()
            os.fsync(f.fileno())
        temp.replace(meta)
        directory = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return ref

    def read(self, ref: ArtifactRef | dict, *, owner: str) -> bytes:
        ref = ArtifactRef.model_validate(ref)
        path = self.root / (ref.id + ".json")
        blob = self.root / (ref.id + ".blob")
        if path.is_symlink() or blob.is_symlink():
            raise ValueError("Artifact symlinks are forbidden")
        stored = ArtifactRef.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if stored.owner != owner or ref.owner != owner:
            raise PermissionError("Artifact belongs to another owner")
        if stored != ref:
            raise ValueError("Artifact reference does not match stored metadata")
        if stored.expires_at is not None and time.time() >= stored.expires_at:
            raise ValueError("Artifact expired")
        if blob.stat().st_size != stored.size or stored.size > self.max_size:
            raise ValueError("Artifact size mismatch")
        data = blob.read_bytes()
        if len(data) != stored.size or len(data) > self.max_size:
            raise ValueError("Artifact size mismatch")
        return data

    def delete(self, ref: ArtifactRef | dict, *, owner: str) -> None:
        """Remove one owned blob, including an expired blob, without following symlinks."""
        ref = ArtifactRef.model_validate(ref)
        meta, blob = self.root / (ref.id + ".json"), self.root / (ref.id + ".blob")
        if meta.is_symlink() or blob.is_symlink():
            raise ValueError("Artifact symlinks are forbidden")
        stored = ArtifactRef.model_validate(json.loads(meta.read_text(encoding="utf-8")))
        if stored.owner != owner or ref.owner != owner:
            raise PermissionError("Artifact belongs to another owner")
        if stored != ref:
            raise ValueError("Artifact reference does not match stored metadata")
        blob.unlink(missing_ok=True)
        meta.unlink()

    def purge_expired(self, *, owner: str) -> int:
        """Explicit owner-scoped maintenance; no implicit deletion during reads."""
        removed = 0
        for path in self.root.glob("*.json"):
            if path.is_symlink():
                continue
            stored = ArtifactRef.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if path.stem != stored.id:
                continue
            if (
                stored.owner == owner
                and stored.expires_at is not None
                and stored.expires_at <= time.time()
            ):
                self.delete(stored, owner=owner)
                removed += 1
        return removed
