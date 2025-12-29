import numpy as np
import math
from config import BETA

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def compute_dpo_loss(best, worst, prompt, policy_model, ref_model):
    lp_best = policy_model.logprob(prompt, best)
    lp_worst = policy_model.logprob(prompt, worst)

    ref_lp_best = ref_model.logprob(prompt, best)
    ref_lp_worst = ref_model.logprob(prompt, worst)

    delta = (lp_best - lp_worst) - (ref_lp_best - ref_lp_worst)
    loss = -math.log(sigmoid(BETA * delta))

    return loss

def compute_grpo_advantages(rewards):
    rewards = np.array(rewards)
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    advantages = (rewards - mean) / std
    return advantages.tolist()
