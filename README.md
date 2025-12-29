# Online RLHF with Preference Optimization (DPO & GRPO)

This project implements a **practical Online RLHF pipeline** focused on **preference-based optimization** under real-world constraints, rather than idealized research assumptions.

The goal is to demonstrate how I reason about:
- Open-ended system design
- Imperfect tooling
- API-based LLM limitations
- Shipping a working solution under ambiguity

This mirrors how ML systems are actually built at early-stage startups.

---

## Why This Project Exists

Most RLHF examples assume:
- Open-weight models
- Token-level log probabilities
- Unlimited compute

In reality, startups often rely on:
- Closed or API-based LLMs
- Limited observability
- Tight iteration timelines

This project intentionally embraces those constraints and shows how to **move forward anyway**.

---

## What This System Does

### End-to-End Flow
1. Generate multiple candidate reasoning traces for a given question  
2. Verify correctness against ground truth  
3. Rank responses using preference signals  
4. Apply preference optimization logic (DPO / GRPO)  
5. Expose everything via a simple Gradio UI  

---

## Core Components

### 1. Verified Trace Generation
- Generates multiple independent responses per question  
- Retains only responses that match the ground-truth answer  
- Ensures preference learning operates on **correct reasoning**, not noise  

This avoids reinforcing confidently wrong outputs—a common RLHF failure mode.

---

### 2. Direct Preference Optimization (DPO)

Implements the full **mathematical structure** of DPO:
- Preferred vs non-preferred response comparison  
- Policy vs reference model contrast  
- Temperature-scaled sigmoid objective  

Because the underlying model does not expose token-level log probabilities, **logprobs are mocked** to demonstrate:
- Correct loss formulation  
- Understanding of how DPO actually works  
- Where real optimization would plug in  

This is intentional and documented.

---

### 3. Group Relative Policy Optimization (GRPO)

Implements:
- Reward normalization  
- Advantage computation  
- Group-relative stabilization  

GRPO is included to demonstrate how preference learning scales beyond pairwise comparisons.

---

## Model Choice & Tradeoffs

### Why Gemini?

I initially attempted the open-source models suggested in the assignment, but they lacked:
- Consistent grounding  
- Stable reasoning across samples  

I switched to **Gemini** to ensure:
- Predictable behavior  
- Cleaner preference comparisons  


This reflects a common startup reality:  
> *The right model does not always expose the right internals.*

---
## Video Walkthrough

The repository includes a short video that covers:

- System architecture
- Design tradeoffs
- RLHF logic
 
----

### If This Were Going to Production

## Given more time or access to open-weight models, I would:

- Swap Gemini for a self-hosted model with logprobs
- Implement true policy updates
- Add online preference collection
- Introduce reward model distillation
- Add monitoring for preference drift

## Repository Structure

```text
.
├── app.py                 # Gradio UI
├── sampler_test.py        # Verified trace generation
├── optimizer.py           # DPO & GRPO loss logic
├── model_interface.py     # Gemini abstraction
├── config.py              # Hyperparameters
├── requirements.txt
├── video_demo.mp4         # Walkthrough and explanation
└── README.md



