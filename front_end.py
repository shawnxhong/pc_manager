import gradio as gr
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def _messages_to_chatbot(messages: list[dict]) -> list[tuple[str, str]]:
    chat_pairs = []
    pending_user = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant":
            chat_pairs.append((pending_user or "", content))
            pending_user = None
    if pending_user:
        chat_pairs.append((pending_user, ""))
    return chat_pairs


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
                model_id = gr.Dropdown(
                    label="Model",
                    choices=llm_choices,
                    value=default_llm,
                    interactive=True,
                )
                device = gr.Dropdown(
                    label="Device",
                    choices=["CPU", "GPU"],
                    value="GPU",
                    interactive=True,
                )
                load_btn = gr.Button("Load Model")
                release_btn = gr.Button("Release Model")

            status = gr.Markdown(value=_format_status(self.pipeline.get_status()))

            chat = gr.Chatbot(
                value=[],
                avatar_images=[user_logo, bot_logo],
                height=720,
                show_copy_button=True,
            )
            working_status = gr.Textbox(
                label="Status",
                value="",
                interactive=False,
                visible=True,
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

            if asr_enabled:
                audio = gr.Audio(
                    label="🎙️ Voice Input",
                    sources=["microphone"],
                    type="filepath",
                    editable=False,
                )
            else:
                audio = None

            def _load_model(selected_model, selected_device):
                info = self.pipeline.ensure_loaded(
                    model_id=selected_model, device=selected_device, no_export=False
                )
                return _format_status(info)

            def _release_model():
                self.pipeline.release()
                return _format_status(self.pipeline.get_status())

            def _submit_message(message, history, progress=gr.Progress(track_tqdm=False)):
                if not message:
                    return "", _messages_to_chatbot(history or []), history, gr.update(value="")
                history = history or []
                history = history + [{"role": "user", "content": message}]
                progress(0, desc="Working...")
                yield "", _messages_to_chatbot(history), history, gr.update(value="Working...")
                try:
                    updated = self.pipeline.run_agent(history)
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
                    yield "", _messages_to_chatbot(updated), updated, gr.update(value="")
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
                    progress(0.5, desc="Working...")
                    yield "", _messages_to_chatbot(streamed), streamed, gr.update(value="Working...")
                progress(1, desc="Done")
                yield "", _messages_to_chatbot(updated), updated, gr.update(value="")

            def _clear_chat():
                return [], [], gr.update(value="")

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
                    return current_input
                try:
                    from pathlib import Path as _Path

                    text = asr_runner(_Path(wav_path))
                    if not isinstance(text, str) or not text.strip():
                        return current_input
                    return text.strip()
                except Exception:
                    return current_input

            load_btn.click(
                _load_model,
                inputs=[model_id, device],
                outputs=[status],
                queue=True,
            )
            release_btn.click(_release_model, outputs=[status], queue=True)

            send_btn.click(
                _submit_message,
                inputs=[user_input, state],
                outputs=[user_input, chat, state, working_status],
                queue=True,
            )
            user_input.submit(
                _submit_message,
                inputs=[user_input, state],
                outputs=[user_input, chat, state, working_status],
                queue=True,
            )
            clear_btn.click(_clear_chat, outputs=[chat, state, working_status], queue=False)
            dump_btn.click(_dump_chat, inputs=[state], outputs=[dump_file], queue=False)

            if audio is not None:
                audio.change(
                    _asr_to_input,
                    inputs=[audio, user_input],
                    outputs=[user_input],
                    queue=False,
                )

        return demo


def agent_demo(agentgr: AgentGradio):
    return agentgr.build_demo()
