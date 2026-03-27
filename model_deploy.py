import modal

# ── 1. Define the container image ──────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # ── All build steps FIRST ──────────────────────────────────────────────
    .apt_install("ffmpeg")
    .pip_install(
        "gradio", "torch", "torchvision",
        "transformers", "matplotlib", "numpy", "fastapi[standard]",
    )
    .run_commands(
        "python -c \"from transformers import AutoTokenizer; "
        "AutoTokenizer.from_pretrained('bert-base-uncased')\""
    )
    # ── Local files LAST (injected at container startup, not build time) ───
    .add_local_python_source("My_model", "Scheduler", "custom_dataset")
    .add_local_file("app.py", "/root/app.py")
    .add_local_file("trained_model100.pt", "/root/trained_model100.pt")
    .add_local_file("Datashape.tsv", "/root/Datashape.tsv")
)

# ── 5. Create the Modal app ─────────────────────────────────────────────────
app = modal.App("diffusion-gradio", image=image)

# ── 6. The serve function ───────────────────────────────────────────────────
@app.function(
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def ui():
    import sys, os
    os.chdir("/root")   # so relative paths like "trained_model100.pt" resolve correctly

    import fastapi
    from gradio.routes import mount_gradio_app
    import gradio as gr

    # Suppress the launch() call inside app.py
    _orig = gr.Interface.launch
    gr.Interface.launch = lambda *a, **kw: None

    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "/root/app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gr.Interface.launch = _orig  # restore

    web_app = fastapi.FastAPI()
    return mount_gradio_app(app=web_app, blocks=module.demo, path="/")
