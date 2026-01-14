#!/usr/bin/env python3
"""Quick test to see if W&B artifact upload is the bottleneck."""

import wandb
import time

print("Initializing W&B...")
run = wandb.init(
    project="specsloeval",
    entity="neuralift-ai",
    name="test-artifact-upload",
    mode="online",
)

print("Creating artifact...")
artifact = wandb.Artifact(name="test-tasks", type="tasks")

print("Adding large file to artifact...")
start = time.time()
artifact.add_file("tasks/hotpot_dev.jsonl")
print(f"File added in {time.time() - start:.2f}s")

print("Logging artifact to W&B...")
start = time.time()
run.log_artifact(artifact)
print(f"Artifact logged in {time.time() - start:.2f}s")

print("Finishing run...")
run.finish()
print("Done!")
