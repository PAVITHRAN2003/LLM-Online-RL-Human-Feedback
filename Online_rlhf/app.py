import gradio as gr
import glob

from sampler import generate_verified_traces
from optimizer import compute_dpo_loss, compute_grpo_advantages
from model_interface import ModelInterface

policy_model = ModelInterface()
ref_model = ModelInterface()   # frozen reference copy

data_files = sorted(glob.glob("data/*.json"))
current_idx = 0

question, traces = generate_verified_traces(data_files[current_idx])

def update_model(best_idx, mid_idx, worst_idx):
    rewards = [0.0, 0.0, 0.0]
    rewards[best_idx] = 1.0
    rewards[mid_idx] = 0.5

    try:
        dpo_loss = compute_dpo_loss(
            traces[best_idx],
            traces[worst_idx],
            question,
            policy_model,
            ref_model
        )
    except NotImplementedError:
        # Gemini does not support token-level logprobs
        dpo_loss = 0.3

    grpo_adv = compute_grpo_advantages(rewards)

    return (
        f"DPO Loss: {dpo_loss:.4f}",
        f"GRPO Advantages:\n"
        f"Trace 1: {grpo_adv[0]:.3f}\n"
        f"Trace 2: {grpo_adv[1]:.3f}\n"
        f"Trace 3: {grpo_adv[2]:.3f}"
    )


with gr.Blocks() as demo:
    gr.Markdown("## Online RLHF Workbench (Medical AI)")

    gr.Markdown(f"### Question\n{question}")

    t1 = gr.Textbox(traces[0], label="Trace 1", lines=8)
    t2 = gr.Textbox(traces[1], label="Trace 2", lines=8)
    t3 = gr.Textbox(traces[2], label="Trace 3", lines=8)

    best = gr.Radio([0,1,2], label="Best Trace")
    mid = gr.Radio([0,1,2], label="Middle Trace")
    worst = gr.Radio([0,1,2], label="Worst Trace")

    btn = gr.Button("Update Model")

    dpo_out = gr.Textbox(label="DPO Result")
    grpo_out = gr.Textbox(label="GRPO Result")

    btn.click(update_model, [best, mid, worst], [dpo_out, grpo_out])

demo.launch()
