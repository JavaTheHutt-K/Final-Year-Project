"""Gradio UI mounted alongside the existing FastAPI backend."""

import os

import gradio as gr
import uvicorn

from backend import api as backend_api


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


def start_experiment():
    response = backend_api.start_run()
    return response, backend_api.get_run_status()


with gr.Blocks(title="Adaptive Intrusion Detection Framework") as demo:
    gr.Markdown("# Adaptive Intrusion Detection Framework")
    gr.Markdown(
        "View experiment status and results, or start the configured "
        "centralised and federated-learning experiment."
    )

    with gr.Row():
        refresh_button = gr.Button("Refresh backend data")
        start_button = gr.Button("Start experiment", variant="primary")
        gpu_check_button = gr.Button("Check GPU availability")
    start_output = gr.JSON(label="Start response")
    gpu_status_output = gr.JSON(label="GPU status")

    with gr.Tab("Status and training history"):
        status_output = gr.JSON(label="Run status")
        history_output = gr.JSON(label="Training history")

    with gr.Tab("Configuration and results"):
        config_output = gr.JSON(label="Experiment configuration")
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
)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
    )
