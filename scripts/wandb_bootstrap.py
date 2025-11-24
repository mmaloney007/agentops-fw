"""Bootstrap a W&B run and log a small table.
Usage:
  WANDB_ENTITY=mike007 WANDB_PROJECT=agentops-fw python scripts/wandb_bootstrap.py
"""
import os, pandas as pd

def main():
    try:
        import wandb
    except Exception as e:
        print("wandb not installed. pip install wandb"); return

    entity = os.getenv("WANDB_ENTITY", "mike007")
    project = os.getenv("WANDB_PROJECT", "agentops-fw")
    wandb.init(project=project, entity=entity, reinit=True, config={"starter": True})

    rows = [
        {"task_id":"summ_001","mode":"constrained","valid":True,"latency_ms":12.3},
        {"task_id":"extract_001","mode":"posthoc","valid":False,"latency_ms":9.8},
    ]
    table = wandb.Table(dataframe=pd.DataFrame(rows))
    wandb.log({"pilot_table": table})
    wandb.finish()
    print("[done] pushed a pilot table to W&B.")

if __name__ == "__main__":
    main()
