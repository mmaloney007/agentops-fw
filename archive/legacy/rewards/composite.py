
from .schema_reward import schema_valid
from .slo_reward import latency_penalty, cost_penalty
from .stability_reward import stability_penalty
def composite_reward(output_json, schema, ok_success:int,
                     latency_ms:float, tokens:int,
                     lam_latency:float, mu_cost:float,
                     disagreement_rate:float=0.0, gamma_stability:float=0.0)->float:
    r = 0.0
    r += 1.0 * schema_valid(output_json, schema)
    r += 1.0 * ok_success
    r += latency_penalty(latency_ms, lam_latency)
    r += cost_penalty(tokens, mu_cost)
    r += stability_penalty(disagreement_rate, gamma_stability)
    return float(r)
