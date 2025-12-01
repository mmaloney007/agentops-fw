
#!/usr/bin/env python3
import os, time, json, pathlib
import wandb
proj=os.getenv("WANDB_PROJECT","agent-stable-slo")
entity=os.getenv("WANDB_ENTITY", None)
mode=os.getenv("WANDB_MODE","offline")
run=wandb.init(project=proj, entity=entity, mode=mode, name="bootstrap")
wandb.log({"ok":1,"ts":time.time()})
path=pathlib.Path("wandb_bootstrap.json")
path.write_text(json.dumps({"ok":1,"ts":time.time()}))
run.finish()
print("[ok] wrote", path)
