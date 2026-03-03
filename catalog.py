# -*- coding: utf-8 -*-
 
from __future__ import annotations
 
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import re
 
 
Domain = Literal["ms_settings", "control_panel", "system_tool"]
 
LaunchKind = Literal[
    "ms_settings_uri",     # value: "ms-settings:display"
    "control_panel_name",  # value: "Microsoft.System"
    "control_panel_cpl",   # value: "powercfg.cpl"
    "control_panel_home",  # value: ""
    "exe",                 # value: "taskmgr.exe"
    "msc",                 # value: "devmgmt.msc"
]
 
@dataclass(frozen=True)
class ToolItem:
    id: str
    domain: Domain
    title: str
    description: str
    launch_kind: LaunchKind
    launch_value: str
    args: Tuple[str, ...] = ()
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "title": self.title,
            "description": self.description,
            "launch": {
                "kind": self.launch_kind,
                "value": self.launch_value,
                "args": list(self.args),
            },
        }
 
 
def _iter_items(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Compatible table may be list[dict] or dict[id->dict].
    Unified output of dict item.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        # It may be {id: {...}} or already a single item
        if "id" in obj and "launch" in obj:
            return [obj]
        items = []
        for k, v in obj.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("id", k)
                items.append(vv)
        return items
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []
 
 
def normalize_catalog(
    ms_settings_items: Any,
    control_panel_items: Any,
    system_tool_items: Any,
) -> List[ToolItem]:
    out: List[ToolItem] = []
 
    def norm_one(x: Dict[str, Any], domain: Domain) -> Optional[ToolItem]:
        tid = (x.get("id") or "").strip()
        title = (x.get("title") or "").strip()
        desc = (x.get("description") or "").strip()
        launch = x.get("launch") or {}
        kind = (launch.get("kind") or "").strip()
        value = (launch.get("value") or "").strip()
        args = launch.get("args") or []
 
        if not tid or not title or not desc or not kind:
            return None
 
        # Normalize kind
        kind_map = {
            "ms_settings": "ms_settings_uri",
            "uri": "ms_settings_uri",
            "control_name": "control_panel_name",
            "name": "control_panel_name",
            "cpl": "control_panel_cpl",
            "control_home": "control_panel_home",
            "home": "control_panel_home",
            "exe": "exe",
            "msc": "msc",
        }
        kind2 = kind_map.get(kind, kind)
 
        if kind2 not in (
            "ms_settings_uri",
            "control_panel_name",
            "control_panel_cpl",
            "control_panel_home",
            "exe",
            "msc",
        ):
            return None
 
        # Auto-fill domain
        # launch_value validation: ms-settings must start with ms-settings:
        if kind2 == "ms_settings_uri":
            if not value.startswith("ms-settings:"):
                return None
 
        # cpl cannot be empty
        if kind2 == "control_panel_cpl" and not value:
            return None
 
        # exe/msc cannot be empty
        if kind2 in ("exe", "msc") and not value:
            return None
 
        # args normalization
        if not isinstance(args, list):
            args = []
        args2 = tuple(str(a) for a in args if a is not None)
 
        return ToolItem(
            id=tid,
            domain=domain,
            title=title,
            description=desc,
            launch_kind=kind2,   # type: ignore
            launch_value=value,
            args=args2,
        )
 
    for x in _iter_items(ms_settings_items):
        item = norm_one(x, "ms_settings")
        if item:
            out.append(item)
 
    for x in _iter_items(control_panel_items):
        item = norm_one(x, "control_panel")
        if item:
            out.append(item)
 
    for x in _iter_items(system_tool_items):
        item = norm_one(x, "system_tool")
        if item:
            out.append(item)
 
    # Deduplication: only keep the first occurrence of each id
    seen = set()
    dedup = []
    for it in out:
        if it.id in seen:
            continue
        seen.add(it.id)
        dedup.append(it)
 
    return dedup
 
 
def build_embedding_text(item: ToolItem) -> str:
    """
    Document text for vector retrieval:
    """
    return (
        f"[{item.domain}] {item.title}\n"
        f"{item.description}\n"
        f"LAUNCH_KIND={item.launch_kind}\n"
        f"LAUNCH_VALUE={item.launch_value}\n"
    )