import os
from pathlib import Path
import requests
from PIL import Image
from typing import Optional
from types import SimpleNamespace

from qwen_agent.llm.schema import CONTENT, ROLE, USER, Message
from qwen_agent.gui.utils import convert_history_to_chatbot
from qwen_agent.gui.gradio_dep import gr, mgr
from qwen_agent.gui import WebUI
import gradio as gr
from questions import QUESTION_POOL
import random
import json
from collections.abc import Iterator

# ---------- 资源 ----------

BASE_DIR = Path(__file__).resolve().parent
bot_logo = BASE_DIR / "bot.jpg"
user_logo = BASE_DIR / "user.jpg"

# ---------- UI 配置 ----------

chatbot_config = {
    "agent.avatar": bot_logo,
    "input.placeholder": "Please input your request here",
    'user.name': "me", 
    "user.avatar": user_logo
}

CHAT_MIN_H = 240
CHAT_MAX_H = 1080000
CHAT_STEP_H = 2400

import qwen_agent.gui.utils as qa_utils

qa_utils.TOOL_CALL = """
<details>
<summary>Searching for solutions...</summary>
</details>
""".strip()

qa_utils.TOOL_OUTPUT = """
<details>
<summary>Finishing this step...</summary>
</details>
""".strip()


class AgentWebUI(WebUI):
    """
    负责构建 Gradio 布局、对接 Qwen-Agent Assistant。
    持有一个 OpenVINOAgentPipeline 实例（通过 pipeline 属性传入）。
    """

    def __init__(self, bot, pipeline, chatbot_config: Optional[dict] = None):
        super().__init__(bot, chatbot_config=chatbot_config or {})
        self.pipeline = pipeline
        self.llm_choices = getattr(pipeline, "llm_choices", None) or []
        self.run_kwargs = {}
        self.footer_links = []

    def _build_chatbot(self, messages):
        from qwen_agent.gui.gradio_dep import mgr

        initial = convert_history_to_chatbot(messages=messages)
        initial = self._scrub_chat_pairs(initial)
        return mgr.Chatbot(
            value=initial,
            avatar_images=[self.user_config, self.agent_config_list],
            height=720,
            avatar_image_width=64,
            flushing=False,
            show_copy_button=True,
            latex_delimiters=[
                {"left": "\\(", "right": "\\)", "display": True},
                {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
                {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
                {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
                {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
                {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
                {"left": "\\[", "right": "\\]", "display": True},
            ],
        )

    def _build_input(self):
        from qwen_agent.gui.gradio_dep import mgr

        return mgr.MultimodalInput(placeholder=self.input_placeholder, elem_id="multimodal-input")

    def _build_asr_audio(self):
        from qwen_agent.gui.gradio_dep import gr

        return gr.Audio(
            label="🎙️ Voice Input",
            sources=["microphone"],
            type="filepath",
            editable=False,
        )
    

    # ---------- ASR ----------

    def _asr_to_input(self, wav_path, current_input):
        """
        使用 pipeline.asr_runner 将语音转文字，并写回 input 组件。
        """
        gr.Info(f"recognizing speech...", duration=2)
        asr_runner_local = getattr(self.pipeline, "asr_runner", None)

        def _extract_triplet(val):
            if isinstance(val, dict):
                return (
                    val.get("text", "") or "",
                    val.get("files", []) or [],
                    val.get("images", []) or [],
                    dict,
                )
            elif isinstance(val, SimpleNamespace):
                return (
                    getattr(val, "text", "") or "",
                    getattr(val, "files", []) or [],
                    getattr(val, "images", []) or [],
                    SimpleNamespace,
                )
            return ("", [], [], dict)

        def _build_like(orig_type, text, files, images):
            if orig_type is SimpleNamespace:
                return SimpleNamespace(text=text, files=files, images=images)
            return {"text": text, "files": files, "images": images}

        text0, files0, images0, orig_type = _extract_triplet(current_input)

        if not asr_runner_local or not wav_path:
            return _build_like(orig_type, text0, files0, images0)

        try:
            from pathlib import Path as _Path

            text = asr_runner_local(_Path(wav_path))
            print(f"[asr] wav: {wav_path}")
            print(f"[asr] text: {text!r}")
            if not isinstance(text, str) or not text.strip():
                return _build_like(orig_type, text0, files0, images0)
            return _build_like(orig_type, text.strip(), [], [])
        except Exception as e:
            print(f"[asr] error: {type(e).__name__}: {e}")
            return _build_like(orig_type, text0, files0, images0)

    # ---------- 构建 Gradio Blocks：不直接 launch，只返回 demo ----------
    def build_demo(
        self,
        messages: Optional[list[Message]] = None,
        concurrency_limit: int = 10,
        enable_mention: bool = False,
    ):
        """
        搭建 Gradio Blocks，不做 launch。
        外部项目可以自行调用 demo.queue().launch(...)。
        """
        from qwen_agent.gui.gradio_dep import gr, ms

        asr_enabled = bool(getattr(self.pipeline, "asr_runner", None))

        llm_choices = getattr(self, "llm_choices", None) or []
        default_llm = llm_choices[0] if llm_choices else None

        with gr.Blocks(
            title="Agentic PC Manager",
            analytics_enabled=False,
            theme=gr.themes.Soft(primary_hue="blue"),
            css="""
            .button-center {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                min-height: 76px !important;
                padding-top: 20px !important;
                background: transparent !important;
            }
            .button-center button {
                margin: 0 !important;
            }
            footer {
                display: none !important;
            }
            .tool-button.upload-button {
                display: none !important;
            }
            """,
        ) as demo:
            MAX_USER_TURNS = 3  # 只保留最近 3 轮对话上下文

            def _get_role(m):
                try:
                    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                except Exception:
                    return None

            def _trim_history_by_user_turns(hist, max_user_turns: int = MAX_USER_TURNS):
                """
                保留：
                - 所有 system（只保留第一个也可以，这里保留全部也行）
                - 从“倒数第 max_user_turns 个 user”开始到结尾的所有消息（含 tool/function）
                并确保截断后的非 system 第一条一定是 user。
                """
                if not isinstance(hist, list):
                    return hist

                sys_msgs = [m for m in hist if _get_role(m) == "system"]
                other = [m for m in hist if _get_role(m) != "system"]

                # 从末尾往前收集，直到收集到 max_user_turns 个 user（包含该 user 及其之后所有消息）
                collected = []
                user_cnt = 0
                for m in reversed(other):
                    collected.append(m)
                    if _get_role(m) == "user":
                        user_cnt += 1
                        if user_cnt >= max_user_turns:
                            break

                kept = list(reversed(collected))

                # 保险：如果开头不是 user，就一直丢到遇到第一个 user（避免 400）
                while kept and _get_role(kept[0]) != "user":
                    kept.pop(0)

                return sys_msgs + kept

            def _trim_ctx(cb, hist):
                return cb, _trim_history_by_user_turns(hist)

            def _backend_status():
                # 单一真相源：后端 pipeline
                if hasattr(self.pipeline, "get_status"):
                    st = self.pipeline.get_status() or {}
                else:
                    st = {"initialized": bool(getattr(self.pipeline, "initialized", False)),
                        "model_id": getattr(self.pipeline, "model_id", None),
                        "device": getattr(self.pipeline, "device", None)}
                return st
            
            def _agent_run_guard(cb, hist):
                try:
                    st = _backend_status()
                    if not st.get("initialized", False):
                        gr.Warning("Model is not loaded. Auto-load may have failed.")
                        yield cb, hist
                        return

                    ret = self.agent_run(cb, hist)
                    if isinstance(ret, Iterator):
                        yield from ret
                    else:
                        yield ret
                except Exception as e:
                    gr.Warning(f"agent_run failed: {type(e).__name__}: {e}")
                    yield cb, hist

            
            def _sync_ui():
                st = _backend_status()
                ready = bool(st.get("initialized", False))
                model_id = st.get("model_id") or "unloaded"
                device = st.get("device") or "-"
            
                llm_md = f"🧠 **Current LLM:** `{model_id if ready else 'unloaded'}`"
                dev_md = f"💻 **Current Device:** `{device}`"
            
                u = gr.update(interactive=ready)
            
                # release 后禁用输入等控件  
                return (llm_md, dev_md, u, u, u, u, u, u)  # 8 个输出

            history = gr.State([])
            backend_status = gr.State({})

            with ms.Application():
                # ---------- 顶部标题 + Release ----------
                with gr.Row():
                    with gr.Column(scale=10):
                        gr.HTML(
                            """
                        <div style="text-align: left; margin-bottom: 10px;">
                            <h1>🤖 Agentic PC Manager</h1>
                        </div>
                        """
                        )
                    with gr.Column(scale=1, min_width=80):
                        btn_release = gr.Button(
                            "🗑️ Release", variant="secondary", size="md", min_width=70
                        )
                        # btn_clear = gr.Button("🧹 Clear Chat", variant="secondary", size="md", min_width=70)

                gr.HTML(
                    """
                <div style="text-align: left; margin-bottom: 10px; font-size: 18px;">
                    Agentic chatbot to bridge your intentions with Windows settings and Windows system tools.
                </div>
                """
                )

                # ---------- 顶部：LLM + Device + Load ----------
                with gr.Row(variant="compact"):
                    dd_llm = gr.Dropdown(
                        choices=llm_choices,
                        value=default_llm,
                        label="LLM",
                        scale=1,
                    )
                    dd_device = gr.Dropdown(
                        choices=["GPU"],
                        value="GPU",
                        label="Device",
                        scale=1,
                    )
                    with gr.Column(scale=1, elem_classes=["button-center"]):
                        btn_load = gr.Button("🔄 Load Model", variant="primary")

                # ---------- 中部：Chat + Input + (ASR) ----------
                with gr.Column(elem_classes="container", scale=1):
                    chatbot = mgr.Chatbot(
                        value=convert_history_to_chatbot(messages=messages or []),
                        avatar_images=[self.user_config, self.agent_config_list],
                        avatar_image_width=64,
                        flushing=False,
                        show_copy_button=True,
                        latex_delimiters=[
                            {"left": "\\(", "right": "\\)", "display": True},
                            {
                                "left": "\\begin{equation}",
                                "right": "\\end{equation}",
                                "display": True,
                            },
                            {
                                "left": "\\begin{align}",
                                "right": "\\end{align}",
                                "display": True,
                            },
                            {
                                "left": "\\begin{alignat}",
                                "right": "\\end{alignat}",
                                "display": True,
                            },
                            {
                                "left": "\\begin{gather}",
                                "right": "\\end{gather}",
                                "display": True,
                            },
                            {
                                "left": "\\begin{CD}",
                                "right": "\\end{CD}",
                                "display": True,
                            },
                            {"left": "\\[", "right": "\\]", "display": True},
                        ],
                        scale=1,
                    )

                    input = mgr.MultimodalInput(
                        placeholder=self.input_placeholder,
                        scale=0,
                        elem_id="pc_manager_input"
                    )
                    # ------------------ pre-defined questions -------------------
                    def _pick_three_questions_list():
                        pool = QUESTION_POOL[:] if QUESTION_POOL else ["Hello"]
                        if len(pool) >= 3:
                            return random.sample(pool, 3)
                        return (pool + pool + pool)[:3]

                    init_q1, init_q2, init_q3 = _pick_three_questions_list()
                    q1_state = gr.State(init_q1)
                    q2_state = gr.State(init_q2)
                    q3_state = gr.State(init_q3)

                    with gr.Row(variant="compact"):
                        btn_q1 = gr.Button(init_q1, variant="secondary", scale=3, interactive=False)
                        btn_q2 = gr.Button(init_q2, variant="secondary", scale=3, interactive=False)
                        btn_q3 = gr.Button(init_q3, variant="secondary", scale=3, interactive=False)
                        btn_refresh_q = gr.Button("🔄 Try others", variant="primary", scale=1, interactive=False)

                    def _pick_three_questions():
                        a, b, c = _pick_three_questions_list()
                        return (
                            a, b, c,
                            gr.update(value=a),
                            gr.update(value=b),
                            gr.update(value=c),
                        )

                    btn_refresh_q.click(
                        fn=_pick_three_questions,
                        inputs=None,
                        outputs=[q1_state, q2_state, q3_state, btn_q1, btn_q2, btn_q3],
                        queue=False,
                    )

                    # --------- ASR ----------
                    if asr_enabled:
                        audio = gr.Audio(
                            label="🎙️ Voice Input",
                            sources=["microphone"],
                            type="filepath",
                            editable=False,
                            scale=0,
                        )
                                
                # ---------- 底部状态：当前 LLM & 设备 ----------
                with gr.Row(variant="compact"):
                    current_llm_md = gr.Markdown(
                        "🧠 **Current LLM:** `unloaded`", container=True
                    )
                    current_device_md = gr.Markdown(
                        "💻 **Current Device:** `-`", container=True
                    )

                # ---------- 底部：Benchmark ----------
                with gr.Accordion("Benchmark", open=True):
                    with gr.Column():
                        bench_in_tokens = gr.Slider(
                            label="Benchmark: input tokens (prompt length)",
                            minimum=64,
                            maximum=2048,
                            step=64,
                            value=256,
                            interactive=True,
                        )
                        bench_out_tokens = gr.Slider(
                            label="Benchmark: output max_new_tokens",
                            minimum=16,
                            maximum=512,
                            step=16,
                            value=128,
                            interactive=True,
                        )
                        bench_btn = gr.Button("📊 Run Benchmark", variant="primary")
                        bench_result = gr.Markdown("**tokens/s:** -")

                # --------- UI lock targets (disable during generation) ----------
                lock_targets = [
                    input,
                    btn_q1, btn_q2, btn_q3, btn_refresh_q,
                    bench_btn, bench_in_tokens, bench_out_tokens,
                    btn_release, btn_load,
                    dd_llm,
                ]
                if asr_enabled:
                    lock_targets.append(audio)

                def _ensure_backend(selected_llm, selected_device):
                    # 这一步在 submit 时触发：自动 load / reload
                    selected_device = "GPU"
                    try:
                        if hasattr(self.pipeline, "ensure_loaded"):
                            st = self.pipeline.ensure_loaded(selected_llm, selected_device)
                        else:
                            # fallback
                            if not getattr(self.pipeline, "initialized", False):
                                self.pipeline.initialize(device=selected_device)
                            st = {"initialized": getattr(self.pipeline, "initialized", False),
                                "model_id": getattr(self.pipeline, "model_id", None),
                                "device": getattr(self.pipeline, "device", None)}
                        return st
                    except Exception as e:
                        gr.Warning(f"Auto-load failed: {type(e).__name__}: {e}")
                        try:
                            return self.pipeline.get_status()
                        except Exception:
                            return {"initialized": False, "model_id": None, "device": None}

                def _apply_status(st):
                    ready = bool((st or {}).get("initialized", False))
                    model_id = (st or {}).get("model_id") or "unloaded"
                    device = (st or {}).get("device") or "-"

                    llm_md = f"🧠 **Current LLM:** `{model_id if ready else 'unloaded'}`"
                    dev_md = f"💻 **Current Device:** `{device}`"

                    u = gr.update(interactive=ready)
                    return (llm_md, dev_md, u, u, u, u, u, u)  # 8 outputs
                
                def _disable_user_controls():
                    # ✅ 输出过程中，禁用：输入框（相当于禁用 submit）、预设问题、换一批、benchmark
                    u = gr.update(interactive=False)
                    return (u, u, u, u, u, u)  # input, q1, q2, q3, refresh_q, bench_btn

                def _enable_user_controls():
                    # ✅ 输出结束后，根据后端真实状态恢复可用性
                    st = _backend_status()
                    ready = bool(st.get("initialized", False))
                    u = gr.update(interactive=ready)
                    return (u, u, u, u, u, u)

                input.submit(
                    fn=self.add_text,
                    inputs=[input, chatbot, history],
                    outputs=[input, chatbot, history],
                    queue=False,
                ).then(
                    fn=_trim_ctx,
                    inputs=[chatbot, history],
                    outputs=[chatbot, history],
                    queue=False,
                ).then(
                    fn=_ensure_backend,
                    inputs=[dd_llm, dd_device],
                    outputs=[backend_status],
                    queue=True,
                ).then(
                    fn=_apply_status,
                    inputs=[backend_status],
                    outputs=[current_llm_md, current_device_md, input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                    queue=False,
                ).then(
                    fn=_disable_user_controls,
                    inputs=None,
                    outputs=[input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                    queue=False,
                ).then(
                    _agent_run_guard,
                    inputs=[chatbot, history],
                    outputs=[chatbot, history],
                ).then(
                    fn=_trim_ctx,
                    inputs=[chatbot, history],
                    outputs=[chatbot, history],
                    queue=False,
                ).then(
                    self.flushed, None, [input]
                ).then(
                    fn=_enable_user_controls,
                    inputs=None,
                    outputs=[input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                    queue=False,
                )


                # ---------- ASR 自动写入 ----------
                if asr_enabled:
                    audio.change(
                        fn=self._asr_to_input,
                        inputs=[audio, input],
                        outputs=[input],
                        queue=False,
                    )

                # --------- random questions -------------
                def _mm_from_text(t: str):
                    return {"text": t, "files": [], "images": []}
                
                def _bind_quick_btn(btn, q_state):
                    btn.click(
                        fn=_mm_from_text,
                        inputs=[q_state],
                        outputs=[input],
                        queue=False,
                    ).then(
                        self.add_text,
                        inputs=[input, chatbot, history],
                        outputs=[input, chatbot, history],
                        queue=False,
                    ).then(
                        fn=_trim_ctx,
                        inputs=[chatbot, history],
                        outputs=[chatbot, history],
                        queue=False,
                    ).then(
                        fn=_ensure_backend,
                        inputs=[dd_llm, dd_device],
                        outputs=[backend_status],
                        queue=True,
                    ).then(
                        fn=_apply_status,
                        inputs=[backend_status],
                        outputs=[current_llm_md, current_device_md, input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                        queue=False,
                    ).then(
                        fn=_disable_user_controls,  # 开始输出前禁用
                        inputs=None,
                        outputs=[input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                        queue=False,
                    ).then(
                        _agent_run_guard,
                        inputs=[chatbot, history],
                        outputs=[chatbot, history],
                    ).then(
                        fn=_trim_ctx,
                        inputs=[chatbot, history],
                        outputs=[chatbot, history],
                        queue=False,
                    ).then(
                        self.flushed, None, [input]
                    ).then(
                        fn=_enable_user_controls,   # 输出结束后恢复
                        inputs=None,
                        outputs=[input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                        queue=False,
                    )

                
                _bind_quick_btn(btn_q1, q1_state)
                _bind_quick_btn(btn_q2, q2_state)
                _bind_quick_btn(btn_q3, q3_state)

                # ---------- 顶部 Load / Release 按钮逻辑 ----------

                def _do_release():
                    try:
                        self.pipeline.release()
                    except Exception as e:
                        gr.Warning(f"release fails: {type(e).__name__}: {e}")
                    gr.Info(f"model released", duration=2)
                    return _sync_ui()

                def _do_load(selected_llm, selected_device):
                    gr.Info(f"loading {selected_llm} on {selected_device}...", duration=5)
                    # selected_llm 仅用于显示/选择；真实状态以 pipeline 为准
                    selected_device = "GPU"
                    if not selected_device:
                        selected_device = "GPU"
                    try:
                        self.pipeline.initialize(device=selected_device)
                    except Exception as e:
                        gr.Warning(f"load fails: {type(e).__name__}: {e}")
                    gr.Info(f"{selected_llm} on {selected_device} loaded", duration=2)
                    return _sync_ui()
                
                btn_release.click(
                    _do_release,
                    inputs=None,
                    outputs=[current_llm_md, current_device_md, input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                    queue=True,
                )
                
                btn_load.click(
                    _do_load,
                    inputs=[dd_llm, dd_device],
                    outputs=[current_llm_md, current_device_md, input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn],
                    queue=True,
                )

                def _do_bench(in_tokens, out_tokens):
                    gr.Info("benchmark executing", duration=5)
                    bench_cb = getattr(self.pipeline, "benchmark", None)
                    if bench_cb is None:
                        gr.Warning("no benchmark backend is provided")
                        return "**tokens/s:** benchmark backend not available"
                
                    ret = bench_cb(int(in_tokens), int(out_tokens))
                
                    if not isinstance(ret, dict) or not ret.get("ok"):
                        err = ret.get("error") if isinstance(ret, dict) else str(ret)
                        gr.Warning(f"benchmark failed: {err}")
                        return f"**tokens/s:** n/a\n\n_error:_ {err}"
                
                    tps = ret.get("tokens_per_s", 0.0)
                    ttft = ret.get("ttft_s", None)
                    tpot = ret.get("tpot_s", None)
                
                    in_tgt = ret.get("input_tokens_target")
                    in_act = ret.get("input_tokens_actual")
                    out_tgt = ret.get("output_tokens_target")
                    warm = ret.get("warmup_iters")
                    iters = ret.get("iters")
                
                    lines = []
                    lines.append(f"**tokens/s:** {tps:.2f}")
                    if ttft is not None:
                        lines.append(f"- TTFT = {ttft*1000:.1f} ms")
                    if tpot is not None:
                        lines.append(f"- TPOT = {tpot*1000:.2f} ms/token")
                    lines.append(f"- input_tokens  (target/actual) = {in_tgt}/{in_act}")
                    lines.append(f"- output_tokens (max_new_tokens)= {out_tgt}")
                    lines.append(f"- warmup_iters = {warm}")
                    lines.append(f"- repeat_iters = {iters}")
                    gr.Info("benchmark finished",duration=2)
                    return "\n".join(lines)

                bench_btn.click(
                    _do_bench,
                    inputs=[bench_in_tokens, bench_out_tokens],
                    outputs=[bench_result],
                    queue=True,
                )
            
            demo.load(
                fn=_sync_ui,
                inputs=None,
                outputs=[current_llm_md, current_device_md, input, btn_q1, btn_q2, btn_q3, btn_refresh_q, bench_btn], 
                queue=False,
            )
            demo.load(
                fn=_pick_three_questions,
                inputs=None,
                outputs=[q1_state, q2_state, q3_state, btn_q1, btn_q2, btn_q3],
                queue=False,
            )

        # 不在这里 queue / launch，由外部项目自己决定
        return demo

    # 不显示工具选择器
    def _create_agent_plugins_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        return gr.Markdown(visible=False)


# ======================================================================
# 对外：Gradio 封装类 + demo 辅助函数
# ======================================================================


class AgentGradio:
    """
    对外暴露的 Gradio 封装类:

    - 持有一个 OpenVINOAgentPipeline 实例（self.pipeline）
    - 提供统一接口：
        - initialize(model_dir, device)
        - release()
        - build_demo() -> gr.Blocks
    """

    def __init__(self, agent_pipeline) -> None:
        self.pipeline = agent_pipeline
        self.multilingual = True  # 与 dummy_example 风格保持一致
        self._ui: AgentWebUI | None = None

    def initialize(
        self,
        model_dir: Optional[str | Path] = None,
        device: str = "GPU",
        no_export: bool = False,
    ):
        """
        对外统一初始化入口。
        外部项目可以选择提前初始化（也可以通过 UI 按钮再初始化）。
        """
        if model_dir is not None:
            # 允许外部覆盖 model_dir
            self.pipeline.model_dir = Path(model_dir)

        if not self.pipeline.initialized:
            self.pipeline.initialize(device=device, no_export=no_export)

    def release(self):
        """
        对外统一释放接口。
        """
        if self.pipeline.initialized:
            self.pipeline.release()

    def build_demo(self, concurrency_limit: int = 10):
        """
        构建 Gradio demo（不 launch）。
        """
        if self._ui is None:
            # 用 pipeline 里已经构造好的 Assistant
            self._ui = AgentWebUI(
                bot=self.pipeline.assistant,
                pipeline=self.pipeline,
                chatbot_config=chatbot_config,
            )
        return self._ui.build_demo(concurrency_limit=concurrency_limit)


def agent_demo(agentgr: AgentGradio):
    return agentgr.build_demo()