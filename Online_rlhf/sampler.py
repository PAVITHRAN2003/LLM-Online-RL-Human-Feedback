import json
import random
import time  # add this
from model_interface import ModelInterface
from config import NUM_CANDIDATES, MAX_NEW_TOKENS
import re

model = ModelInterface()

def extract_final_answer(text):
    """
    Robustly extract final MCQ answer (A/B/C/D) from model output.
    """
    # Common patterns seen in LLM outputs
    patterns = [
        r"\bAnswer\s*[:\-]?\s*([ABCD])\b",
        r"\bThe correct answer is\s*\*?\*?([ABCD])",
        r"\bThe final answer is\s*\$?\\boxed\{?([ABCD])\}?",
        r"\b([ABCD])\)\s",          # B) Something
        r"\*\*([ABCD])\)",          # **B)
        r"^\s*([ABCD])\b"           # starts with B
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    return None

def generate_verified_traces(sample_path):
    with open(sample_path) as f:
        sample = json.load(f)

    question = sample["Questions"]
    gt = sample["Answer"]

    verified = []

    attempts = 0
    while len(verified) < 3 and attempts < NUM_CANDIDATES:
        prompt = (
            f"You are a medical reasoning assistant. For the following multiple-choice question, "
            f"choose **exactly one option (A, B, C, or D)** that is correct. Display the correct option "
            f"followed by Explaination your reasoning in 3–5 sentences **focused only on the question context**. "
            f"Do not add extra disclaimers or platform descriptions.\nQuestion:{question}"
        )

        out = model.generate(prompt, MAX_NEW_TOKENS)
        print(out)

        final = extract_final_answer(out)
        if final == gt:
            verified.append(out)

        attempts += 1

        # Sleep 1–2 seconds between API calls to reduce rate
        time.sleep(25)

    if len(verified) < 3:
        raise RuntimeError("Could not generate 3 verified traces")

    random.shuffle(verified)
    return question, verified
