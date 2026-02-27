import gradio as gr
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from questions import QUESTION_POOL

BASE_DIR = Path(__file__).resolve().parent
bot_logo = BASE_DIR / "bot.jpg"
user_logo = BASE_DIR / "user.jpg"

chatbot_config = {
    "agent.avatar": bot_logo,
    "input.placeholder": "Please input your request here",
    "user.name": "me",
    "user.avatar": user_logo,
}


def _format_status(status: dict) -> str:
    if not status:
        return "Model: not loaded"
    if not status.get("initialized"):
        return "Model: not loaded"
    return f"Model: {status.get('model_id')} | Device: {status.get('device')}"


def _messages_to_chatbot(messages: list[dict]) -> list[dict]:
    chat_messages = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = msg.get("content", "")
        payload = {"role": role, "content": content}
        metadata = msg.get("metadata")
        if metadata:
            payload["metadata"] = metadata
        chat_messages.append(payload)
    return chat_messages


def _pick_suggestions():
    return random.sample(QUESTION_POOL, min(3, len(QUESTION_POOL)))


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

    def initialize(
        self,
        model_dir: Optional[str | Path] = None,
        device: str = "GPU",
        no_export: bool = False,
    ):
        if model_dir is not None:
            self.pipeline.model_dir = Path(model_dir)

        if not self.pipeline.initialized:
            self.pipeline.initialize(device=device, no_export=no_export)

    def release(self):
        if self.pipeline.initialized:
            self.pipeline.release()

    def build_demo(self, concurrency_limit: int = 10):
        llm_choices = getattr(self.pipeline, "llm_choices", []) or []
        default_llm = llm_choices[0] if llm_choices else None
        asr_enabled = bool(getattr(self.pipeline, "asr_runner", None))
        dump_dir = BASE_DIR / "chat_dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)

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
            """,
        ) as demo:
            state = gr.State([])

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
                    release_btn = gr.Button(
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

            with gr.Row():
                model_id = gr.Dropdown(
                    label="Model",
                    choices=llm_choices,
                    value=default_llm,
                    interactive=True,
                    scale=1,
                )
                device = gr.Dropdown(
                    label="Device",
                    choices=["CPU", "GPU"],
                    value="GPU",
                    interactive=True,
                    scale=1,
                )
                with gr.Column(scale=1, elem_classes=["button-center"]):
                    load_btn = gr.Button("Load Model", variant="primary")

            status = gr.Markdown(value=_format_status(self.pipeline.get_status()))

            chat = gr.Chatbot(
                value=[],
                type="messages",
                avatar_images=[user_logo, bot_logo],
                height=720,
                show_copy_button=True,
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder=chatbot_config["input.placeholder"],
                    label="",
                    lines=2,
                )
                send_btn = gr.Button("Send")

            with gr.Row():
                clear_btn = gr.Button("Clear")
                dump_btn = gr.Button("Dump Chat JSON")
            dump_file = gr.File(label="Chat Dump", interactive=False)

            initial_suggestions = _pick_suggestions()
            suggestions_state = gr.State(initial_suggestions)
            with gr.Row():
                q_btn_1 = gr.Button(initial_suggestions[0], variant="secondary", size="sm")
                q_btn_2 = gr.Button(initial_suggestions[1], variant="secondary", size="sm")
                q_btn_3 = gr.Button(initial_suggestions[2], variant="secondary", size="sm")
                refresh_btn = gr.Button("🔄", variant="secondary", size="sm", min_width=50)

            if asr_enabled:
                audio = gr.Audio(
                    label="🎙️ Voice Input",
                    sources=["microphone"],
                    type="filepath",
                    editable=False,
                )
                with gr.Row():
                    asr_btn = gr.Button("Transcribe Voice")
                asr_status = gr.Markdown(value="")
            else:
                audio = None
                asr_btn = None
                asr_status = None

            def _load_model(selected_model, selected_device):
                info = self.pipeline.ensure_loaded(
                    model_id=selected_model, device=selected_device, no_export=False
                )
                return _format_status(info)

            def _release_model():
                self.pipeline.release()
                return _format_status(self.pipeline.get_status())

            def _submit_message(message, history):
                if not message:
                    return "", _messages_to_chatbot(history or []), history
                history = history or []
                history = history + [{"role": "user", "content": message}]
                pending_display = history + [
                    {"role": "assistant", "content": "thinking for solutions..."}
                ]
                yield "", _messages_to_chatbot(pending_display), history
                updated = None
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.pipeline.run_agent, history)
                    while not future.done():
                        yield "", _messages_to_chatbot(pending_display), history
                        time.sleep(0.2)
                    try:
                        updated = future.result()
                    except Exception as exc:
                        updated = history + [
                            {"role": "assistant", "content": f"Error: {exc}"}
                        ]

                assistant_index = None
                for idx in range(len(updated) - 1, -1, -1):
                    if updated[idx].get("role") == "assistant":
                        assistant_index = idx
                        break

                if assistant_index is None:
                    yield "", _messages_to_chatbot(updated), updated
                    return
                full_text = str(updated[assistant_index].get("content", ""))
                partial = ""
                for ch in full_text:
                    partial += ch
                    streamed = list(updated)
                    streamed[assistant_index] = {
                        **updated[assistant_index],
                        "content": partial,
                    }
                    yield "", _messages_to_chatbot(streamed), streamed
                yield "", _messages_to_chatbot(updated), updated

            def _clear_chat():
                return [], []

            def _dump_chat(history: list[dict]) -> str | None:
                payload = {
                    "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "messages": history or [],
                    "debug": {
                        "status": self.pipeline.get_status(),
                        "model_dir": str(getattr(self.pipeline, "model_dir", "")),
                        "llm_config": getattr(self.pipeline, "llm_cfg", None),
                        "tools": [
                            getattr(tool, "name", str(tool))
                            for tool in (getattr(self.pipeline, "tools", None) or [])
                        ],
                    },
                }
                safe_payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
                filename = datetime.utcnow().strftime("chat_dump_%Y%m%d_%H%M%S.json")
                path = dump_dir / filename
                path.write_text(
                    json.dumps(safe_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return str(path)

            def _asr_to_input(wav_path, current_input):
                asr_runner = getattr(self.pipeline, "asr_runner", None)
                if not asr_runner or not wav_path:
                    return current_input, "⚠️ No voice input detected."
                try:
                    from pathlib import Path as _Path

                    text = asr_runner(_Path(wav_path))
                    if not isinstance(text, str) or not text.strip():
                        return current_input, "⚠️ Could not transcribe voice. Please try again."
                    text = text.strip()
                    return text, f"✅ Transcribed: {text}"
                except Exception as exc:
                    return current_input, f"❌ ASR failed: {exc}"

            # -- Collect all interactive buttons for disable/enable pattern --
            all_btns = [
                send_btn, clear_btn, dump_btn, load_btn, release_btn,
                q_btn_1, q_btn_2, q_btn_3, refresh_btn,
            ]
            if asr_btn is not None:
                all_btns.append(asr_btn)

            def _disable_btns():
                return [gr.update(interactive=False)] * len(all_btns)

            def _enable_btns():
                return [gr.update(interactive=True)] * len(all_btns)

            def _refresh_suggestions():
                new = _pick_suggestions()
                return (
                    new,
                    gr.update(value=new[0]),
                    gr.update(value=new[1]),
                    gr.update(value=new[2]),
                )

            def _make_question_handler(idx):
                def _handler(suggestions, history):
                    yield from _submit_message(suggestions[idx], history)
                return _handler

            # -- Wire events with disable/enable pattern --

            load_btn.click(
                _disable_btns, outputs=all_btns
            ).then(
                _load_model, inputs=[model_id, device], outputs=[status], queue=True
            ).then(
                _enable_btns, outputs=all_btns
            )

            release_btn.click(
                _disable_btns, outputs=all_btns
            ).then(
                _release_model, outputs=[status], queue=True
            ).then(
                _enable_btns, outputs=all_btns
            )

            send_btn.click(
                _disable_btns, outputs=all_btns
            ).then(
                _submit_message, inputs=[user_input, state],
                outputs=[user_input, chat, state], queue=True
            ).then(
                _enable_btns, outputs=all_btns
            )

            user_input.submit(
                _disable_btns, outputs=all_btns
            ).then(
                _submit_message, inputs=[user_input, state],
                outputs=[user_input, chat, state], queue=True
            ).then(
                _enable_btns, outputs=all_btns
            )

            clear_btn.click(
                _disable_btns, outputs=all_btns
            ).then(
                _clear_chat, outputs=[chat, state]
            ).then(
                _enable_btns, outputs=all_btns
            )

            dump_btn.click(
                _disable_btns, outputs=all_btns
            ).then(
                _dump_chat, inputs=[state], outputs=[dump_file]
            ).then(
                _enable_btns, outputs=all_btns
            )

            for idx, q_btn in enumerate([q_btn_1, q_btn_2, q_btn_3]):
                q_btn.click(
                    _disable_btns, outputs=all_btns
                ).then(
                    _make_question_handler(idx),
                    inputs=[suggestions_state, state],
                    outputs=[user_input, chat, state], queue=True
                ).then(
                    _enable_btns, outputs=all_btns
                )

            refresh_btn.click(
                _disable_btns, outputs=all_btns
            ).then(
                _refresh_suggestions,
                outputs=[suggestions_state, q_btn_1, q_btn_2, q_btn_3]
            ).then(
                _enable_btns, outputs=all_btns
            )

            if audio is not None:
                audio.change(
                    _asr_to_input,
                    inputs=[audio, user_input],
                    outputs=[user_input, asr_status],
                    queue=False,
                )
                asr_btn.click(
                    _disable_btns, outputs=all_btns
                ).then(
                    _asr_to_input,
                    inputs=[audio, user_input],
                    outputs=[user_input, asr_status]
                ).then(
                    _enable_btns, outputs=all_btns
                )

        return demo


def agent_demo(agentgr: AgentGradio):
    return agentgr.build_demo()
