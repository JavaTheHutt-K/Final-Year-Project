"""Gradio UI mounted alongside the existing FastAPI backend."""

import os

import gradio as gr
import uvicorn

from backend import api as backend_api
from backend.auth import verify_gradio_admin, verify_gradio_credentials


def _optional_env(name: str, default: str | None = None) -> str | None:
    """Return a trimmed environment value or the fallback when blank."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def check_gpu_availability():
    import torch

    return {"cuda_available": torch.cuda.is_available()}


def refresh_backend_state():
    return (
        backend_api.get_run_status(),
        backend_api.get_config(),
        backend_api.get_results(),
        backend_api.get_training_history(),
    )


def start_experiment(request: gr.Request):
    verify_gradio_admin(request.username)
    response = backend_api.start_run_for_authorized_admin()
    return response, backend_api.get_run_status()


dashboard_css = """
:root {
    --ink: #e8eef8;
    --muted: #91a0b7;
    --line: rgba(148, 163, 184, 0.16);
    --panel: rgba(15, 23, 42, 0.72);
    --panel-strong: #111c31;
    --accent: #61d9c2;
    --accent-soft: rgba(97, 217, 194, 0.13);
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto;
    padding: 28px 24px 54px !important;
    background:
        radial-gradient(circle at 8% 0%, rgba(52, 211, 153, 0.11), transparent 28rem),
        radial-gradient(circle at 94% 8%, rgba(59, 130, 246, 0.12), transparent 28rem);
}

.hero {
    padding: 34px 38px 32px;
    margin-bottom: 22px;
    border: 1px solid var(--line);
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(17, 28, 49, 0.96), rgba(15, 23, 42, 0.78));
    box-shadow: 0 20px 60px rgba(2, 8, 23, 0.28);
}

.eyebrow {
    margin: 0 0 10px;
    color: var(--accent);
    font: 700 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0;
    color: var(--ink);
    font-size: clamp(2rem, 4vw, 3.25rem);
    letter-spacing: -0.045em;
    line-height: 1.05;
}

.hero-copy {
    max-width: 680px;
    margin: 15px 0 0;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.65;
}

.section-heading {
    margin: 26px 0 12px;
    color: var(--ink);
    font-size: 17px;
    font-weight: 650;
}

.action-bar {
    padding: 16px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.58);
}

.action-bar button {
    min-height: 44px;
    border-radius: 10px !important;
    font-weight: 650 !important;
}

.action-bar button.primary {
    box-shadow: 0 8px 22px rgba(97, 217, 194, 0.2);
}

.output-card {
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    background: var(--panel) !important;
    overflow: hidden;
}

.output-card > div:first-child {
    padding: 13px 16px 8px;
}

.output-card .label-wrap {
    color: var(--muted);
    font-size: 12px;
    font-weight: 650;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.tab-nav {
    gap: 8px;
    margin-top: 24px;
    border-bottom: 1px solid var(--line);
}

.tab-nav button {
    border: 0 !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
}

.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

@media (max-width: 700px) {
    .gradio-container { padding: 16px 12px 36px !important; }
    .hero { padding: 26px 22px; border-radius: 17px; }
}
"""

with gr.Blocks(
    title="Adaptive Intrusion Detection Framework",
    theme=gr.themes.Base(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    ),
    css=dashboard_css,
) as demo:
    gr.HTML(
        """
        <header class="hero">
            <p class="eyebrow">Security research console · live workspace</p>
            <h1>Adaptive Intrusion<br>Detection Framework</h1>
            <p class="hero-copy">
                Monitor experiment health, review training progress, and launch
                centralised or federated-learning runs from one focused control centre.
            </p>
        </header>
        """
    )

    gr.Markdown("### Experiment controls", elem_classes="section-heading")
    with gr.Row(elem_classes="action-bar"):
        refresh_button = gr.Button("↻  Refresh data")
        start_button = gr.Button("▶  Start experiment", variant="primary")
        gpu_check_button = gr.Button("▣  Check GPU")

    with gr.Row():
        with gr.Column(elem_classes="output-card"):
            start_output = gr.JSON(label="Latest launch response")
        with gr.Column(elem_classes="output-card"):
            gpu_status_output = gr.JSON(label="Compute availability")

    with gr.Tabs():
        with gr.Tab("Live status", id="status"):
            with gr.Row():
                with gr.Column(elem_classes="output-card"):
                    status_output = gr.JSON(label="Run status")
                with gr.Column(elem_classes="output-card"):
                    history_output = gr.JSON(label="Training history")
        with gr.Tab("Experiment data", id="data"):
            with gr.Row():
                with gr.Column(elem_classes="output-card"):
                    config_output = gr.JSON(label="Experiment configuration")
                with gr.Column(elem_classes="output-card"):
                    results_output = gr.JSON(label="Experiment results")

    refresh_button.click(
        refresh_backend_state,
        outputs=[
            status_output,
            config_output,
            results_output,
            history_output,
        ],
    )
    start_button.click(
        start_experiment,
        outputs=[start_output, status_output],
    )
    gpu_check_button.click(
        check_gpu_availability,
        outputs=gpu_status_output,
    )

    timer = gr.Timer(5)
    timer.tick(
        lambda: (
            backend_api.get_run_status(),
            backend_api.get_training_history(),
        ),
        outputs=[status_output, history_output],
    )


app = gr.mount_gradio_app(
    backend_api.app,
    demo,
    path="/",
    ssr_mode=False,
    auth=(
        verify_gradio_credentials
        if _optional_env("GRADIO_ADMIN_USERNAME")
        and _optional_env("ADMIN_PASSWORD_HASH")
        else None
    ),
)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(_optional_env("PORT") or "7860"),
    )
