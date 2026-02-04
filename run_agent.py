#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from qwen_agent.llm import get_chat_model
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
from tools import get_langgraph_tools
from pc_manager_prompt import PC_MANAGER_SYSTEM_PROMPT
import gc
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- 常量：脚本目录 & 默认模型目录 -----------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 模型根目录：project_root/model
MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_OV_DIR = MODEL_ROOT / "Qwen3-4B-Instruct-ov"
DEFAULT_HF_REPO = "shawnxhong/Qwen3-4B-Instruct-ov"


# ----------------- IR 获取：从 HuggingFace 直接拉取 -----------------


def ensure_openvino_ir(repo_id: str, model_dir: Path):
    if model_dir.exists():
        return

    print(f"[prepare] {model_dir} not found, pulling from HuggingFace {repo_id} ……")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "no huggingface_hub found, please install: pip install huggingface_hub"
        )

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        revision="main",
    )

    print(f"[prepare] pulled model from HuggingFace to {model_dir}")


# ----------------- LLM 构建 -----------------
def build_llm_cfg(model_path: Path, device: str, fast_kv: bool):
    if fast_kv:
        ov_config = {
            "KV_CACHE_PRECISION": "u8",
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
            hints.performance_mode(): hints.PerformanceMode.LATENCY,
            streams.num(): "",
            props.cache_dir(): "",
        }
    else:
        ov_config = {
            hints.performance_mode(): hints.PerformanceMode.LATENCY,
            streams.num(): "1",
            props.cache_dir(): "",
        }

    llm_cfg = {
        "ov_model_dir": str(model_path),
        "model_type": "openvino",
        "device": device,
        "ov_config": ov_config,
        "generate_cfg": {
                            "do_sample": False,
                            "fncall_prompt_type": "qwen",
                            "renormalize_logits": True,
                            "remove_invalid_values": True,
                            "max_new_tokens": 512,
                            "max_time": 30.0,
                            "repetition_penalty": 1.08,
                            # "max_input_tokens": 4096,
                        },
    }
    return llm_cfg


def build_llm(model_path: Path, device: str, fast_kv: bool):
    llm_cfg = build_llm_cfg(model_path, device, fast_kv)
    llm = get_chat_model(llm_cfg)
    return llm, llm_cfg


# ----------------- 外部 ASR -----------------
def run_asr(
    exe_path: Path,
    model_path: Path,
    wav_path: Path,
    timeout_s: int = 120,
) -> str:
    if not exe_path.exists():
        raise FileNotFoundError(f"ASR exe not found: {exe_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"ASR model not found: {model_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"wav file not found: {wav_path}")

    proc = subprocess.run(
        [str(exe_path), "-m", str(model_path), "-w", str(wav_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ASR exit code {proc.returncode}\nSTDERR:\n{proc.stderr}")

    lines = []
    tag_re = re.compile(r"<\|[^>]+?\|>")
    for raw in proc.stdout.splitlines():
        raw_s = raw.lstrip()

        if any(
            raw_s.startswith(p)
            for p in (
                "[",
                "OpenVINO",
                "Plugin info",
                "Active code page",
                "CPU SNIPPETS_MODE",
                "VAD",
                "Successfully",
                "Read model took",
                "Version :",
                "Build   :",
            )
        ):
            continue

        if "Version :" in raw or "Build   :" in raw:
            continue

        clean = tag_re.sub("", raw).strip()
        if clean:
            lines.append(clean)
    transcript = " ".join(lines).strip()
    return transcript


# ----------------- 占位 LLM -----------------


class _UnloadedLLM:
    """
    启动/释放后的占位 LLM：未真正加载模型时使用。
    """

    def __init__(self):
        self.model = "unloaded-llm"
        self.model_type = "openvino_unloaded"

    def chat(self, *a, **k):
        raise RuntimeError(
            "LLM is not loaded. please select Device and press Load Model。"
        )


class AgentState(TypedDict):
    messages: list[dict]
    tool_request: Optional[dict]


def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        for item in reversed(response):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", ""))
        return " ".join(_extract_text(item) for item in response)
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, list):
            return " ".join(str(x) for x in content)
        if content is not None:
            return str(content)
        return json.dumps(response, ensure_ascii=False)
    return str(response)


def _parse_tool_request(text: str) -> Optional[dict]:
    match = re.search(
        r"Action\s*:\s*(?P<name>[\w\-]+)\s*Action Input\s*:\s*(?P<input>.+)",
        text,
        re.S,
    )
    if not match:
        return None
    name = match.group("name").strip()
    raw_input = match.group("input").strip()
    raw_input = raw_input.split("\\nFinal:", 1)[0].strip()
    raw_input = raw_input.split("\\nObservation:", 1)[0].strip()
    if raw_input.startswith("```"):
        raw_input = raw_input.strip("`").strip()
    if raw_input.endswith(".") and "{" in raw_input and "}" in raw_input:
        raw_input = raw_input[:-1].strip()
    try:
        args = json.loads(raw_input)
    except json.JSONDecodeError:
        match_json = re.search(r"(\{.*\})", raw_input, re.S)
        if match_json:
            try:
                args = json.loads(match_json.group(1))
            except json.JSONDecodeError:
                args = {"input": raw_input}
        else:
            args = {"input": raw_input}
    return {"name": name, "args": args}


def _build_tool_system_prompt(tools) -> str:
    tool_lines = [
        "You can call tools using the following format:",
        "Action: tool_name",
        "Action Input: {json}",
        "",
    ]
    tool_lines.append("Available tools:")
    for tool in tools:
        description = getattr(tool, "description", "") or ""
        args = getattr(tool, "args", None)
        args_json = json.dumps(args, ensure_ascii=False) if args else "{}"
        tool_lines.append(f"- {tool.name}: {description} Args: {args_json}")
    tool_lines.append("")
    tool_lines.append(
        "When you respond to the user, do NOT include Action/Action Input/Final tags. "
        "Only output the final user-facing response."
    )
    return "\n".join(tool_lines)


class LangGraphAgentRunner:
    def __init__(self, llm, tools, system_prompt: str):
        self.llm = llm
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.system_prompt = f"{system_prompt}\n\n{_build_tool_system_prompt(tools)}"
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("assistant", self._assistant_node)
        graph.add_node("tool", self._tool_node)

        def route(state: AgentState):
            if state.get("tool_request"):
                return "tool"
            return END

        graph.add_conditional_edges("assistant", route, {"tool": "tool", END: END})
        graph.add_edge("tool", "assistant")
        graph.set_entry_point("assistant")
        return graph.compile()

    def _assistant_node(self, state: AgentState):
        messages = state.get("messages", [])
        followup = bool(messages and messages[-1].get("role") == "tool")
        chat_messages = messages
        if followup:
            chat_messages = messages + [
                {
                    "role": "system",
                    "content": (
                        "Use the tool result to decide the next step. "
                        "If more tools are needed (e.g., ambiguous candidates), call them. "
                        "Otherwise respond to the user without Action/Action Input/Final tags."
                    ),
                }
            ]
        response = self.llm.chat(messages=chat_messages, stream=False)
        text = _extract_text(response)
        tool_request = _parse_tool_request(text)
        if tool_request:
            return {
                "messages": messages,
                "tool_request": tool_request,
            }
        return {
            "messages": messages + [{"role": "assistant", "content": text}],
            "tool_request": None,
        }

    def _tool_node(self, state: AgentState):
        tool_request = state.get("tool_request")
        messages = state.get("messages", [])
        if not tool_request:
            return {"messages": messages, "tool_request": None}

        name = tool_request.get("name")
        args = tool_request.get("args") or {}
        tool = self.tool_map.get(name)
        if tool is None:
            result = json.dumps(
                {"ok": False, "error": f"unknown tool: {name}"},
                ensure_ascii=False,
            )
        else:
            try:
                result = tool.invoke(args)
            except Exception as exc:
                result = json.dumps(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                )
        return {
            "messages": messages + [{"role": "tool", "content": result, "name": name}],
            "tool_request": None,
        }

    def invoke(self, messages: list[dict], recursion_limit: int = 6) -> list[dict]:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        result = self._graph.invoke(
            {"messages": messages, "tool_request": None},
            config={"recursion_limit": recursion_limit},
        )
        return result["messages"]


# ----------------- 统一流水线类：OpenVINOAgentPipeline -----------------
class OpenVINOAgentPipeline:
    """
    对外暴露的统一接口：
      - initialized: 是否已经完成模型加载
      - initialize(device, no_export): 加载/切换模型
      - release(): 释放模型、清理资源
      - benchmark(target_tokens): 简单 tokens/s 测试

    内部持有：
      - self.llm / self.llm_cfg
      - self.agent_runner (LangGraph agent)
      - self.tools (LangGraph tools)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_HF_REPO,
        model_dir: Optional[Path] = None,
        fast_kv: bool = False,
        asr_model: Optional[Path] = None, 
        asr_device: str = "CPU", 
        asr_type: str = "sensevoice"
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_OV_DIR
        self.fast_kv = fast_kv
        self.device: Optional[str] = None

        self.llm = None
        self.llm_cfg: Optional[Dict[str, Any]] = None
        self.tools = get_langgraph_tools()
        self.agent_runner = LangGraphAgentRunner(
            llm=_UnloadedLLM(),
            tools=self.tools,
            system_prompt=PC_MANAGER_SYSTEM_PROMPT,
        )

        # ASR (ai_speech backend)
        self.asr_engine = None
        if asr_model:
            try:
                from asr_ai_speech_runner import AISpeechASRRunner
                self.asr_engine = AISpeechASRRunner(model_dir=Path(asr_model), 
                                                    device=asr_device, 
                                                    model_type=asr_type)
                self.asr_runner = self.asr_engine  # callable(wav_path)->text
            except Exception as e:
                print(f"[ASR] disabled: {type(e).__name__}: {e}")
                self.asr_engine = None
                self.asr_runner = None
        else:
            self.asr_runner = None

        self._lock = threading.RLock()
        self.initialized: bool = False

    @property
    def llm_choices(self):
        return [self.model_id]

    # ---- 内部：释放 LLM 资源 ----
    def _hard_drop_llm(self, llm):
        try:
            for attr in [
                "compiled_model",
                "core",
                "tokenizer",
                "pipeline",
                "ov_model",
                "_compiled_model",
                "_core",
                "_tokenizer",
                "_pipeline",
                "_ov_model",
                "generation_config",
                "streamer",
                "_infer_req",
                "kv_cache",
            ]:
                if hasattr(llm, attr):
                    try:
                        setattr(llm, attr, None)
                    except Exception:
                        pass
            for fn in ("unload", "close", "teardown", "shutdown"):
                m = getattr(llm, fn, None)
                if callable(m):
                    try:
                        m()
                    except Exception:
                        pass
        except Exception:
            pass

    # ---- 对外：初始化 / 加载 ----

    def initialize(self, device: str = "GPU", no_export: bool = False) -> None:
        """
        准备/加载 OpenVINO LLM；多次调用是幂等的。
        如 device 不同，会自动 release 后重新加载。
        """
        # 已经有模型且 device 一样，直接返回
        with self._lock:
            if self.initialized and self.device == device:
                return

            # device 改变：先释放旧模型
            if self.initialized and self.device != device:
                self.release()

            # 确保 IR 在本地
            if not no_export:
                ensure_openvino_ir(self.model_id, self.model_dir)

            # 构造 LLM
            llm, llm_cfg = build_llm(self.model_dir, device, self.fast_kv)
            self.llm = llm
            self.llm_cfg = llm_cfg
            self.device = device

            if self.agent_runner is not None:
                self.agent_runner.llm = llm

            self.initialized = True
            gc.collect()

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前模型状态。
        """
        status = {
            "initialized": self.initialized,
            "model_id": (self.model_id if self.initialized else None),
            "device": (self.device if self.initialized else None),
        }
        return status

    def _resolve_model_dir(self, model_id: str) -> Path:
        """Try best-effort to map model_id -> local OpenVINO IR dir under MODEL_ROOT."""
        name = (model_id.split("/")[-1] if model_id else "").strip()
        if not name:
            return self.model_dir

        candidates = [
            MODEL_ROOT / name,
            MODEL_ROOT / f"{name}-ov",
            MODEL_ROOT / f"{name}_ov",
            MODEL_ROOT / model_id.replace("/", "__"),
        ]
        for p in candidates:
            if p.exists():
                return p

        # default target dir if none exists yet
        return MODEL_ROOT / f"{name}-ov"

    
    def ensure_loaded(self, model_id: Optional[str], device: str, no_export: bool = False) -> Dict[str, Any]:
        """
        Ensure backend is loaded with (model_id, device).
        - If not loaded -> load.
        - If loaded but device/model mismatch -> reload.
        Returns current status dict.
        """
        device = (device or "GPU").upper()
        with self._lock:
            desired_model = model_id or self.model_id

            # If model differs, update model_id & model_dir
            if desired_model and desired_model != self.model_id:
                self.model_id = desired_model
                self.model_dir = self._resolve_model_dir(desired_model)

            need_reload = (not self.initialized) or (self.device != device)
            if need_reload:
                # If initialized but device mismatch -> release first
                if self.initialized:
                    try:
                        self.release()
                    except Exception:
                        pass

                # Ensure IR then initialize
                try:
                    if not no_export:
                        ensure_openvino_ir(self.model_id, self.model_dir)
                    self.initialize(device=device, no_export=True)  # IR already ensured above
                except TypeError:
                    # your initialize signature is initialize(device, no_export)
                    self.initialize(device=device, no_export=no_export)

            return self.get_status()


    # ---- 对外：释放 ----

    def release(self) -> None:
        """
        释放当前模型资源，并将 Assistant 恢复为占位 LLM。
        """
        with self._lock:
            llm = self.llm or getattr(self.agent_runner, "llm", None)
            if llm is not None and not isinstance(llm, _UnloadedLLM):
                self._hard_drop_llm(llm)

            if self.agent_runner is not None:
                self.agent_runner.llm = _UnloadedLLM()

            self.llm = None
            self.llm_cfg = None
            self.device = None
            self.initialized = False
            gc.collect()

    def release_all(self) -> None:
        """
        释放当前模型资源，并将 Assistant 恢复为占位 LLM。
        同时, 释放RAG 资源。
        """
        with self._lock:
            try:
                self.release()
            except Exception:
                pass
            try:
                from tools import release_rag_resource
                release_rag_resource()
            except Exception:
                pass

            try:
                if getattr(self, "asr_engine", None) is not None:
                    self.asr_engine.release()
            except Exception:
                pass
            self.asr_runner = None
            self.asr_engine = None

            gc.collect()

    # ---- 对外：benchmark ----
    def benchmark(self, input_tokens: int = 256, output_tokens: int = 128) -> Dict[str, Any]:
        from collections.abc import Iterator
        from time import perf_counter as _now
        import math
    
        WARMUP_ITERS = 1
        REPEAT_ITERS = 1
    
        if self.llm is None or not self.initialized or isinstance(self.llm, _UnloadedLLM):
            return {"ok": False, "error": "LLM is not initialized"}
    
        # ---------- helpers ----------
        def _extract_text(chunk) -> str:
            if chunk is None:
                return ""
            if isinstance(chunk, str):
                return chunk
            if isinstance(chunk, dict):
                c = chunk.get("content", "")
                if isinstance(c, list):
                    return " ".join(_extract_text(x) for x in c)
                return str(c) if c is not None else ""
            if isinstance(chunk, list):
                return " ".join(_extract_text(x) for x in chunk)
            return str(chunk)
    
        def _find_tokenizer(llm):
            for attr in ("tokenizer", "_tokenizer"):
                tok = getattr(llm, attr, None)
                if tok is not None:
                    return tok
            ov = getattr(llm, "ov_model", None) or getattr(llm, "_ov_model", None)
            if ov is not None:
                for attr in ("tokenizer", "_tokenizer"):
                    tok = getattr(ov, attr, None)
                    if tok is not None:
                        return tok
            return None
    
        tok = _find_tokenizer(self.llm)
    
        def _count_delta_tokens(text: str) -> int:
            if not text:
                return 0
            if tok is None:
                return len(text.split())
            try:
                return len(tok.encode(text, add_special_tokens=False))
            except Exception:
                return len(text.split())
    
        def _count_prompt_tokens(messages) -> int | None:
            if tok is None:
                return None
            try:
                if hasattr(tok, "apply_chat_template"):
                    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    if isinstance(ids, dict) and "input_ids" in ids:
                        ids = ids["input_ids"]
                    if isinstance(ids, (list, tuple)):
                        return len(ids)
            except Exception:
                return None
            return None
    
        def _build_prompt_for_target_input_tokens(target_tokens: int) -> tuple[str, int, str]:
            base = "Benchmark input. Please answer briefly.\n"
            filler = "hello "
    
            if tok is None:
                n = max(0, target_tokens - len(base.split()))
                content = base + (filler * n)
                return content, len(content.split()), "approx_words"
    
            def tokens_for_n(n: int) -> int:
                content = base + (filler * n)
                ct = _count_prompt_tokens([{"role": "user", "content": content}])
                return ct if ct is not None else len(content.split())
    
            lo, hi = 0, 20000
            best_text = base
            best_tokens = tokens_for_n(0)
            method = "tokenizer+chat_template"
    
            # 拉 hi 到超过 target
            while hi < 20000 and tokens_for_n(hi) < target_tokens:
                hi = min(20000, hi * 2)
    
            # 二分逼近
            while lo <= hi:
                mid = (lo + hi) // 2
                t = tokens_for_n(mid)
                if abs(t - target_tokens) < abs(best_tokens - target_tokens):
                    best_tokens = t
                    best_text = base + (filler * mid)
                if t < target_tokens:
                    lo = mid + 1
                else:
                    hi = mid - 1
    
            return best_text, best_tokens, method
    
        input_tokens = int(max(1, input_tokens))
        output_tokens = int(max(1, output_tokens))
    
        user_content, actual_in_tokens, in_method = _build_prompt_for_target_input_tokens(input_tokens)
        messages = [{"role": "user", "content": user_content}]
    
        # ---------- warmup ----------
        warm_cfg = {"max_new_tokens": 8, "do_sample": False}
        for _ in range(WARMUP_ITERS):
            try:
                _ = self.llm.chat(messages=messages, stream=False, generate_cfg=warm_cfg)
            except TypeError:
                _ = self.llm.chat(messages=messages)
    
        # ---------- measured ----------
        gen_cfg = {"max_new_tokens": int(output_tokens), "do_sample": False}
    
        details = []
        sum_ttft = 0.0
        sum_total = 0.0
        sum_out_tokens = 0
        sum_tpot = 0.0
        tpot_cnt = 0
    
        for i in range(REPEAT_ITERS):
            t0 = _now()
            first_t = None
    
            out_tokens_i = 0
            accum_text = ""   # 用于识别“累积全文 chunk”
            last_good = ""    # 兜底记录最后一次的 cur_text
    
            try:
                resp = self.llm.chat(messages=messages, stream=True, generate_cfg=gen_cfg)
            except TypeError:
                resp = self.llm.chat(messages=messages)
    
            if isinstance(resp, Iterator):
                for chunk in resp:
                    cur = _extract_text(chunk) or ""
                    if not cur:
                        continue
    
                    last_good = cur
    
                    # 关键：只算增量 delta
                    if cur.startswith(accum_text):
                        delta = cur[len(accum_text):]
                        accum_text = cur
                    elif accum_text.startswith(cur):
                        # 有些实现会偶尔回退更短的文本，忽略
                        delta = ""
                    else:
                        # 不可判定是累积还是增量：保守当作增量
                        delta = cur
                        accum_text += cur
    
                    dtok = _count_delta_tokens(delta)
                    if dtok > 0 and first_t is None:
                        first_t = _now()
                    out_tokens_i += dtok
            else:
                # 非 streaming：只能把整段当作输出
                cur = _extract_text(resp) or ""
                last_good = cur
                out_tokens_i = _count_delta_tokens(cur)
                first_t = None  # TTFT 不可得，用 total 近似
    
            t1 = _now()
            total = max(t1 - t0, 1e-6)
            ttft = (first_t - t0) if first_t is not None else total
    
            # TPOT
            tpot = None
            if out_tokens_i >= 2 and total > ttft:
                tpot = (total - ttft) / max(1, out_tokens_i - 1)
                if math.isfinite(tpot):
                    sum_tpot += tpot
                    tpot_cnt += 1
    
            sum_ttft += ttft
            sum_total += total
            sum_out_tokens += out_tokens_i
    
            details.append({
                "iter": i,
                "ttft_s": ttft,
                "total_s": total,
                "out_tokens": out_tokens_i,
                "tpot_s": tpot,
            })
    
        avg_ttft = sum_ttft / max(1, REPEAT_ITERS)
        avg_tpot = (sum_tpot / tpot_cnt) if tpot_cnt else None
        tps = (sum_out_tokens / sum_total) if sum_total > 0 else 0.0
    
        return {
            "ok": True,
            "input_tokens_target": input_tokens,
            "input_tokens_actual": actual_in_tokens,
            "input_tokens_method": in_method,
            "output_tokens_target": output_tokens,
            "warmup_iters": WARMUP_ITERS,
            "iters": REPEAT_ITERS,
    
            "ttft_s": float(avg_ttft),
            "tpot_s": (float(avg_tpot) if avg_tpot is not None else None),
            "tokens_per_s": float(tps),
    
            "time_s": float(sum_total),
            "out_tokens_total": int(sum_out_tokens),
            "details": details,
            "first_token_s": float(avg_ttft),  # 兼容你前端旧字段
        }

    def chat(self, messages, **generate_cfg):
        """
        直接用当前 LLM 调 chat（不经过 Gradio UI）。
        """
        if not self.initialized or self.llm is None:
            raise RuntimeError("LLM is not initialized")

        if generate_cfg:
            return self.llm.chat(messages=messages, generate_cfg=generate_cfg)
        return self.llm.chat(messages=messages)

    def run_agent(self, messages: list[dict]) -> list[dict]:
        """
        Run the LangGraph agent with the current message history.
        """
        if not self.initialized or self.llm is None:
            raise RuntimeError("LLM is not initialized")
        if self.agent_runner is None:
            raise RuntimeError("Agent runner is not initialized")
        return self.agent_runner.invoke(messages)

    def run_agent(self, messages: list[dict]) -> list[dict]:
        """
        Run the LangGraph agent with the current message history.
        """
        if not self.initialized or self.llm is None:
            raise RuntimeError("LLM is not initialized")
        if self.agent_runner is None:
            raise RuntimeError("Agent runner is not initialized")
        return self.agent_runner.invoke(messages)


# ----------------- main：CLI 启动 Gradio UI -----------------


def main():
    parser = argparse.ArgumentParser(
        description="Function-calling Agent with OpenVINO + LangGraph (OOP wrapper)"
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_HF_REPO,
        help="HuggingFace 模型仓库名（例如 shawnxhong/Qwen3-4B-Instruct-ov）",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_OV_DIR),
        help="OpenVINO IR 本地目录，默认放在 run_agent.py 同级的 Qwen3-4B-Instruct-ov/",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        choices=["GPU"],
        help="推理设备",
    )
    parser.add_argument(
        "--fast-kv", action="store_true", help="启用 KV-cache/激活量化等加速配置"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="跳过从 HuggingFace 拉取",
    )
    parser.add_argument("--port", type=int, default=7861, help="Gradio 端口")
    parser.add_argument("--share", action="store_true", help="Gradio 共享外网")       
    parser.add_argument(
        "--asr_model",
        type=Path,
        default=None,
        help="ASR model directory for ai_speech backend",
    )
    parser.add_argument(
        "--asr_device",
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="ASR device (default CPU)",
    )
    parser.add_argument(
        "--asr_type",
        default="sensevoice",
        help="ASR model type for ai_speech (default sensevoice)",
    )

    args = parser.parse_args()

    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

    model_dir = Path(args.model_path)

    # 构造统一流水线对象（此处先不主动 initialize，交给 UI 里的 Load 按钮触发）
    asr_model = args.asr_model
    if not asr_model:
        asr_model = None
        print("ASR not enabled: model path not provided")

    pipeline = OpenVINOAgentPipeline(
        model_id=args.model_id,
        model_dir=model_dir,
        fast_kv=args.fast_kv,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        asr_type=args.asr_type,
    )

    # UI 封装 + demo 构建
    from front_end import AgentGradio, agent_demo

    gradio_helper = AgentGradio(pipeline)
    demo = agent_demo(gradio_helper)

    # 注意：真正的 initialize 由 UI 顶部的 Load Model 按钮触发
    demo.queue(default_concurrency_limit=1).launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
