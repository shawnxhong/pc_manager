import gradio as gr
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
                    choices=["GPU"],
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

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder=chatbot_config["input.placeholder"],
                    label="",
                    lines=2,
                )
                send_btn = gr.Button("Send")

            with gr.Row():
                clear_btn = gr.Button("Clear")

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

            def _append_user_message(message, history):
                if not message:
                    return "", _messages_to_chatbot(history or []), history
                history = history or []
                updated_history = history + [{"role": "user", "content": message}]
                return "", _messages_to_chatbot(updated_history), updated_history

            def _stream_response(history):
                history = history or []
                try:
                    updated = self.pipeline.run_agent(history)
                except Exception as exc:
                    updated = history + [{"role": "assistant", "content": f"Error: {exc}"}]

                response_text = ""
                for msg in reversed(updated):
                    if msg.get("role") == "assistant":
                        response_text = msg.get("content", "")
                        break
                if not isinstance(response_text, str):
                    response_text = str(response_text)

                base_pairs = _messages_to_chatbot(history)
                if history and history[-1].get("role") == "user":
                    user_text = history[-1].get("content", "")
                    if base_pairs:
                        base_pairs = base_pairs[:-1]
                else:
                    user_text = ""

                partial = ""
                for ch in response_text:
                    partial += ch
                    yield base_pairs + [(user_text, partial)], history

                yield _messages_to_chatbot(updated), updated

            def _clear_chat():
                return [], []

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
                _append_user_message,
                inputs=[user_input, state],
                outputs=[user_input, chat, state],
                queue=False,
            )
            send_btn.then(
                _stream_response,
                inputs=[state],
                outputs=[chat, state],
                queue=True,
            )
            user_input.submit(
                _append_user_message,
                inputs=[user_input, state],
                outputs=[user_input, chat, state],
                queue=False,
            )
            user_input.then(
                _stream_response,
                inputs=[state],
                outputs=[chat, state],
                queue=True,
            )
            clear_btn.click(_clear_chat, outputs=[chat, state], queue=False)

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
