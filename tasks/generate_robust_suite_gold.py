#!/usr/bin/env python3
"""
Generate a gold-labeled robust evaluation suite (~500 tasks) with schemas and
answer keys for accuracy/faithfulness/citation scoring.
Outputs: tasks/robust_eval_gold.jsonl
"""
import json, random, pathlib
random.seed(123)
ROOT = pathlib.Path(__file__).parent
SCHEMAS = {
    "summary": "tasks/schemas/summary_schema.json",
    "summary_cite": "tasks/schemas/summary_cite_schema.json",
    "contact": "tasks/schemas/extract_contact_schema.json",
    "tool": "tasks/schemas/tool_plan_schema.json",
    "math": "tasks/schemas/math_steps_schema.json",
    "code": "tasks/schemas/code_stub_schema.json",
    "safety": "tasks/schemas/safety_schema.json",
    "toolseq": "tasks/schemas/tool_sequence_schema.json",
    "summary_multi": "tasks/schemas/summary_schema.json",
    "qa": "tasks/schemas/qa_schema.json",
    "helmsmall": "tasks/schemas/qa_schema.json"
}

def mk_summary(id_suffix: int, title: str, bullets: list[str], sources: list[str]):
    src_lines=[]
    for idx,s in enumerate(sources):
        txt = bullets[idx % len(bullets)] if bullets else f"Fact {idx}"
        src_lines.append(f"[{s}]: {txt}")
    prompt = (
        f"Sources:\n" + "\n".join(src_lines) + "\n"
        f"Write a title and 3-6 bullets summarizing: {title}. Cite sources using [id] tokens."
    )
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["summary_cite"],
        "gold": {"title": title, "bullets": bullets, "citations": sources},
        "id": f"summary_{id_suffix}"
    }

def mk_extraction(id_suffix: int, name: str, email: str, company: str, role: str):
    prompt = f"Extract contact info (name/email/company/role) from the following text: {name} ({role}) at {company}; email {email}."
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["contact"],
        "gold": {"name": name, "email": email, "company": company, "role": role},
        "id": f"contact_{id_suffix}"
    }

def mk_tool(id_suffix: int, tool: str, inputs: dict, expected_output: str, deadline: int):
    prompt = f"Plan a tool call '{tool}' with inputs {inputs}; expected output: {expected_output}; deadline {deadline} minutes."
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["tool"],
        "gold": {"tool": tool, "inputs": inputs, "expected_output": expected_output, "deadline_minutes": deadline},
        "id": f"tool_{id_suffix}"
    }

def mk_toolseq(id_suffix: int, steps: list[str]):
    prompt = "Plan the minimal tool sequence (1-3 steps) to accomplish the task. Return steps[]."
    return {
        "prompt": prompt + f" Task: {steps[-1]}",
        "schema_path": SCHEMAS["toolseq"],
        "gold": {"steps": steps},
        "id": f"toolseq_{id_suffix}"
    }

def mk_multiturn(id_suffix: int, user_turns: list[str]):
    convo = "\n".join([f"User: {u}" for u in user_turns])
    prompt = f"Conversation:\n{convo}\nSummarize key requests as title + bullets."
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["summary_multi"],
        "gold": {"title": f"Multi-turn {id_suffix}", "bullets": user_turns[:3]},
        "id": f"multiturn_{id_suffix}"
    }

def mk_longcontext(id_suffix: int, base: str):
    long_text = " ".join([base]*50)
    prompt = f"Long context:\n{long_text}\nSummarize the main point as title + bullets."
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["summary"],
        "gold": {"title": base[:30], "bullets": [base[:50]]},
        "id": f"longctx_{id_suffix}"
    }

def mk_qa(id_suffix: int, question: str, answer: str):
    prompt = f"Question: {question}\nAnswer succinctly."
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["qa"],
        "gold": {"answer": answer},
        "id": f"qa_{id_suffix}"
    }

def mk_gaia_math(id_suffix: int):
    q=f"What is {id_suffix}+3?"
    a=str(id_suffix+3)
    return mk_qa(id_suffix, q, a)

def mk_math(id_suffix: int, desc: str, answer: float):
    prompt = f"Solve and show steps: {desc}"
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["math"],
        "gold": {"answer": answer},
        "id": f"math_{id_suffix}"
    }

def mk_safety(id_suffix: int, decision: str, reason: str, policy: str, risk: int):
    prompt = f"Review this action for policy '{policy}'. Decision? {reason}"
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["safety"],
        "gold": {"decision": decision, "reason": reason, "policy": policy, "risk_score": risk},
        "id": f"safety_{id_suffix}"
    }

def mk_code(id_suffix: int, func: str, signature: str, doc: str, tests: list[str]):
    prompt = f"Provide a stub: {func} with signature {signature}. Add docstring and 1-3 tests."
    return {
        "prompt": prompt,
        "schema_path": SCHEMAS["code"],
        "gold": {"function": func, "signature": signature, "docstring": doc, "tests": tests},
        "id": f"code_{id_suffix}"
    }

def truth_stress():
    cases=[]
    scenarios=[
        ("CTR inversion", ["CTR lower than population despite higher spend", "Lift efforts ongoing"], ["s1","s2"]),
        ("Churn vs NPS mismatch", ["Churn is higher; NPS also higher", "Retention plan pending"], ["s3","s4"]),
        ("Latency vs throughput", ["Throughput higher; p95 latency worse", "Queueing effects observed"], ["s5","s6"]),
        ("Security incidents", ["Incidents up post-mitigation", "Additional controls needed"], ["s7","s8"]),
        ("Compliance status", ["Audit findings increased", "Compliance debt noted"], ["s9","s10"]),
    ]
    for i,(title,bullets,sources) in enumerate(scenarios):
        cases.append(mk_summary(200+i, title, bullets, sources))
    return cases

def main():
    tasks=[]
    # Summaries with citations
    for i in range(50):
        tasks.append(mk_summary(i, f"Segment {i} QBR", [f"Metric-{i} improved", f"Risk-{i} noted", f"Action-{i} assigned"], ["s1","s2"]))
    # Extraction
    names=["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Heidi","Ivan","Judy"]
    for i in range(80):
        n=names[i % len(names)]
        tasks.append(mk_extraction(i, f"{n} Q.{i}", f"{n.lower()}.{i}@example.org", "Acme Corp", "Engineer"))
    # Tool plans
    for i in range(80):
        tasks.append(mk_tool(i, "search_contracts", {"keyword":"SLA","clause_id":f"C{i}"}, "List matching clauses", deadline=30))
    # Math
    for i in range(60):
        tasks.append(mk_math(i, f"Compute compound growth for principal 1000, rate 5%, years {i%5+1}", 1000*((1.05)**(i%5+1))))
    # Safety
    for i in range(60):
        decision="flag" if i%3==0 else "allow"
        tasks.append(mk_safety(i, decision, "PII detected" if decision=="flag" else "No PII", "data_privacy", 4 if decision=="flag" else 1))
    # Tool sequencing
    seqs=[["search_contracts", "summarize_report"], ["classify_intent"], ["run_sql_query", "file_ticket"]]
    for i in range(20):
        tasks.append(mk_toolseq(i, seqs[i % len(seqs)]))
    # Multi-turn
    for i in range(20):
        tasks.append(mk_multiturn(i, [f"Turn {i} request", "Follow-up detail", "Constraint: respond in JSON"]))
    # Long-context
    for i in range(20):
        tasks.append(mk_longcontext(i, f"Key insight {i} repeated"))
    # GAIA/HELM-like QA/math slice
    for i in range(30):
        tasks.append(mk_gaia_math(i))
    # Code (lightweight scoring)
    for i in range(40):
        tasks.append(mk_code(i, f"fn_p95_{i}", "def fn_p95(latencies):", "Compute p95 from list", ["fn_p95([1,2,3])"]))
    # Truth stress
    tasks.extend(truth_stress())
    # Add extras to reach ~500
    while len(tasks) < 580:
        j=len(tasks)
        tasks.append(mk_summary(j, f"General summary {j}", [f"Point A{j}", f"Point B{j}", f"Point C{j}"], ["s1","s2"]))
    out_path = ROOT / "robust_eval_gold.jsonl"
    with open(out_path,"w",encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False)+"\n")
    print(f"[done] wrote {len(tasks)} tasks to {out_path}")

if __name__=="__main__":
    main()
