## Online RLHF Workbench

This project simulates an Online RLHF pipeline for medical reasoning.

### Flow
1. Generate multiple reasoning traces
2. Filter by ground truth correctness
3. Human ranks correct traces (Best/Mid/Worst)
4. System computes:
   - DPO Loss
   - GRPO Advantages

### Run
```bash
pip install -r requirements.txt
python app.py
