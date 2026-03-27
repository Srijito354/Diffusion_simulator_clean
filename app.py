import tempfile

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer

from My_model import Diffusion_model
from Scheduler import Noise_scheduler
from custom_dataset import Data

# ----------- Setup (loaded once) -----------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Diffusion_model().to(device)
model.load_state_dict(torch.load("trained_model100.pt", map_location=device))
model.eval()

scheduler = Noise_scheduler(device=device)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

dataset = Data("Datashape.tsv")

mean = dataset.mean
std = dataset.std


@torch.no_grad()
def render_reverse_diffusion(prompt: str) -> str:
    """Run reverse diffusion and return path to an MP4 video file."""

    if prompt is None or prompt.strip() == "":
        raise gr.Error("Please enter a non-empty prompt.")

    prompt = prompt.strip()

    if prompt not in dataset.label_point_dict:
        raise gr.Error(
            f"Prompt not found in dataset labels. Available labels: {', '.join(sorted(dataset.label_point_dict.keys()))}"
        )

    # Tokenize prompt
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Start from noise
    x_t = torch.randn(1, 142, 2).to(device)

    # Real shape
    real_points = np.array(dataset.label_point_dict[prompt])

    # Figure layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    scatter_gen = ax1.scatter([], [], s=10, c="tomato")
    scatter_real = ax2.scatter(real_points[:, 0], real_points[:, 1], s=10, c="steelblue")

    ax1.set_title("Reverse Diffusion")
    ax2.set_title("Real Shape")

    ax1.axis("equal")
    ax2.axis("equal")

    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)

    title = fig.suptitle("")

    # consistent axis limits
    all_pts = np.concatenate([real_points])

    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)

    margin = 0.3
    xr = xmax - xmin
    yr = ymax - ymin

    xmin -= margin * xr
    xmax += margin * xr
    ymin -= margin * yr
    ymax += margin * yr

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)

    @torch.no_grad()
    def update(frame):
        nonlocal x_t

        t = scheduler.T - frame - 1
        t_tensor = torch.tensor([t]).to(device)

        predicted_noise = model(x_t, t_tensor, input_ids, attention_mask)

        x_t = scheduler.reverse_step(x_t, t_tensor, predicted_noise)

        pts = x_t.squeeze(0).cpu().numpy()
        pts = pts * std + mean

        scatter_gen.set_offsets(pts)

        title.set_text(f"Reverse Diffusion | timestep {t}")

        return scatter_gen,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=scheduler.T,
        interval=20,
        blit=False,
    )

    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    writer = animation.FFMpegWriter(fps=30, bitrate=1800)
    ani.save(output_path, writer=writer, dpi=200)

    plt.close(fig)

    return output_path


demo = gr.Interface(
    fn=render_reverse_diffusion,
    inputs=gr.Textbox(label="Enter shape", lines=1, placeholder="e.g. circle"),
    outputs=gr.Video(label="Reverse Diffusion Video"),
    title="Diffusion Simulator",
    description="Enter a shape label and watch the reverse diffusion process reconstruct the shape.",
)


if __name__ == "__main__":
    demo.launch(share = True)
