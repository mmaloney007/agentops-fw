
def stability_penalty(disagreement_rate: float, gamma: float) -> float:
    return -gamma * disagreement_rate
