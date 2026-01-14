
def latency_penalty(latency_ms: float, lam: float) -> float:
    return -lam * (latency_ms / 1000.0)
def cost_penalty(tokens: int, mu: float) -> float:
    return -mu * (tokens / 1000.0)
