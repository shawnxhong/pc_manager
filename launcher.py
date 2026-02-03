# -*- coding: utf-8 -*-
 
from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess 
from catalog import ToolItem, ToolItem, normalize_catalog
from typing import Any, Dict, List, Optional
from rag_index import Embedder, ToolRAGIndex

 
@dataclass
class LaunchResult:
    ok: bool
    cmdline: List[str]
    pid: Optional[int] = None
    error: Optional[str] = None
 
 
def build_cmdline(item: ToolItem) -> List[str]:
    k = item.launch_kind
    v = item.launch_value
    args = list(item.args or ())
 
    if k == "ms_settings_uri":
        # 用 cmd start 打开 URI（最稳）
        return ["cmd", "/c", "start", "", v]
 
    if k == "control_panel_home":
        return ["control.exe"]
 
    if k == "control_panel_name":
        # Control Panel canonical name
        return ["control.exe", "/name", v]
 
    if k == "control_panel_cpl":
        # .cpl
        return ["control.exe", v]
 
    if k == "msc":
        # 统一走 mmc.exe，避免 WinError 193
        return ["mmc.exe", v] + args
 
    if k == "exe":
        return [v] + args
 
    # 理论上不会到这里
    return ["cmd", "/c", "start", "", v]
 
 
def launch(item: ToolItem) -> LaunchResult:
    cmd = build_cmdline(item)
 
    try:
        creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
        proc = subprocess.Popen(
            cmd,
            shell=False,
            creationflags=creation_flags,
            cwd=str(Path.cwd()),
        )
        return LaunchResult(ok=True, cmdline=cmd, pid=getattr(proc, "pid", None))
    except Exception as e:
        return LaunchResult(ok=False, cmdline=cmd, error=f"{type(e).__name__}: {e}")
    
 
 
@dataclass
class ResolveOutput:
    ok: bool
    intent: str
    domain: Optional[str]
    target_id: Optional[str]
    score: float
    candidates: List[Dict[str, Any]]
    reason: Optional[str] = None
    error: Optional[str] = None
 
 
class PCManager:
    def __init__(
        self,
        ms_settings_items: Any,
        control_panel_items: Any,
        system_tool_items: Any,
        index_dir: Path,
        embedder_model: str = "BAAI/bge-m3",
        embedder_device: str = "cpu",
    ):
        self.catalog: List[ToolItem] = normalize_catalog(
            ms_settings_items, control_panel_items, system_tool_items
        )
        self.id_to_item = {it.id: it for it in self.catalog}
 
        self.embedder = Embedder(model_name=embedder_model, device=embedder_device)
        self.index = ToolRAGIndex(self.catalog, index_dir=index_dir, embedder=self.embedder)
        self.index.load_or_build()
 
    def resolve(self, intent: str, top_k: int = 3) -> ResolveOutput:
        try:
            cands = self.index.search(intent, top_k=top_k)
            payload = []
            for c in cands:
                it = self.id_to_item.get(c.tool_id)
                if not it:
                    continue
                payload.append({
                    "domain": it.domain,
                    "target_id": it.id,
                    "score": c.score,
                    "title": it.title,
                })
 
            if not payload:
                return ResolveOutput(
                    ok=False, intent=intent,
                    domain=None, target_id=None, score=0.0,
                    candidates=[],
                    error="no candidates"
                )
 
            best = payload[0]
            return ResolveOutput(
                ok=True,
                intent=intent,
                domain=best["domain"],
                target_id=best["target_id"],
                score=float(best["score"]),
                candidates=payload,
            )
        except Exception as e:
            return ResolveOutput(
                ok=False, intent=intent,
                domain=None, target_id=None, score=0.0,
                candidates=[],
                error=f"{type(e).__name__}: {e}"
            )
 
    def open(
        self,
        intent: str,
        reason: Optional[str] = None,
        top_k: int = 3,
        min_score: float = 0.34,
        min_margin: float = 0.005,
        target_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 允许 agent “指定打开某个候选项”，彻底消除二义性
        if target_id:
            it = self.id_to_item.get(target_id)
            if not it:
                return {
                    "ok": False,
                    "intent": intent,
                    "reason": reason,
                    "error": f"unknown target_id: {target_id}",
                }
            lr = launch(it)
            return {
                "ok": lr.ok,
                "intent": intent,
                "reason": reason,
                "domain": it.domain,
                "target_id": it.id,
                "cmdline": lr.cmdline,
                "pid": lr.pid,
                "error": lr.error,
                "score": None,
                "candidates": None,
            }
 
        r = self.resolve(intent, top_k=top_k)
        if not r.ok or not r.candidates:
            return {
                "ok": False,
                "intent": intent,
                "reason": reason,
                "error": r.error or "resolve failed",
                "candidates": r.candidates,
            }
 
        # 歧义判断：分数太低或 top1/top2 太接近就不自动打开
        best = r.candidates[0]
        score1 = float(best["score"])
        score2 = float(r.candidates[1]["score"]) if len(r.candidates) > 1 else -1.0
 
        if score1 < float(min_score) or (score2 >= 0 and (score1 - score2) < float(min_margin)):
            return {
                "ok": False,
                "intent": intent,
                "reason": reason,
                "error": "ambiguous_or_low_confidence",
                "domain": best["domain"],
                "target_id": best["target_id"],
                "score": score1,
                "candidates": r.candidates,
            }
 
        it = self.id_to_item[best["target_id"]]
        lr = launch(it)
        return {
            "ok": lr.ok,
            "intent": intent,
            "reason": reason,
            "domain": it.domain,
            "target_id": it.id,
            "score": score1,
            "candidates": r.candidates,
            "cmdline": lr.cmdline,
            "pid": lr.pid,
            "error": lr.error,
        }
    
    def close(self) -> None:
        """
        release rag resource, embedder, index
        """
        try:
            if getattr(self, "index", None) is not None:
                self.index.close()
        except Exception:
            pass

        try:
            if getattr(self, "embedder", None) is not None:
                self.embedder.close()
        except Exception:
            pass