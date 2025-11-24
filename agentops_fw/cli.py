import argparse
from .core import run_tasks

def main():
    p = argparse.ArgumentParser(description="AgentOps-FW: contract-grounded agents (single GPU).")
    p.add_argument("--tasks", default="tasks/pilot.json")
    p.add_argument("--mode", choices=["posthoc","constrained","contracts","budgeted"], default="posthoc")
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--project", default="agentops-fw", help="W&B project name")
    args = p.parse_args()
    run_tasks(args.tasks, args.mode, args.out, project=args.project)

if __name__ == "__main__":
    main()
