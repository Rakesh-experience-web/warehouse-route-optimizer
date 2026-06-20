from __future__ import annotations

import json
import os
import time
import uuid
from typing import List

from app.schemas import MapLayoutPayload, StoredMap


class LayoutStore:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def _ensure_parent(self) -> None:
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)

    def _load(self) -> List[StoredMap]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [StoredMap.model_validate(item) for item in raw]

    def _save(self, items: List[StoredMap]) -> None:
        self._ensure_parent()
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in items], f, indent=2)

    def list_maps(self) -> List[StoredMap]:
        return self._load()

    def get_map(self, map_id: str) -> StoredMap | None:
        for m in self._load():
            if m.map_id == map_id:
                return m
        return None

    def save_map(self, payload: MapLayoutPayload) -> StoredMap:
        items = self._load()
        stored = StoredMap(
            map_id=str(uuid.uuid4()),
            name=payload.name,
            layout=payload.layout,
            shelf_categories=payload.shelf_categories,
            created_at_epoch=int(time.time()),
        )
        items.insert(0, stored)
        self._save(items)
        return stored

