
#!/usr/bin/env python3
import os, sys, subprocess, itertools, pathlib
LAMBDAS=[0.0,0.1,0.2]; MUS=[0.0,0.05,0.1]; GAMMAS=[0.0]
STEPS=int(os.getenv("SWEEP_STEPS","400"))
PREFIX=os.getenv("SWEEP_PREFIX","out/sweeps")
BASE_MODEL=os.getenv("SWEEP_BASE_MODEL","Qwen/Qwen2.5-7B-Instruct")
def run(cmd): print(">>"," ".join(cmd)); subprocess.run(cmd, check=True)
def main():
    pathlib.Path(PREFIX).mkdir(parents=True, exist_ok=True)
    for lam, mu, gam in itertools.product(LAMBDAS, MUS, GAMMAS):
        os.environ["LAMBDA_LATENCY"]=str(lam)
        os.environ["MU_COST"]=str(mu)
        os.environ["GAMMA_STABILITY"]=str(gam)
        out=f"{PREFIX}/lam{lam}_mu{mu}_gam{gam}"
        run([sys.executable,"-m","agent_stable_slo.train.grpo_trl",
             "--base-model",BASE_MODEL,
             "--tasks","tasks/fc_tasks.jsonl",
             "--out",out,"--steps",str(STEPS),"--max-new-tokens","196"])
if __name__=="__main__": main()
