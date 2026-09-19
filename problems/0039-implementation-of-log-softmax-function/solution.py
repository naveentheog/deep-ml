import math

def log_softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    log_total = math.log(sum(math.exp(s - m) for s in scores))  # sum >= 1, so log is safe
    return [(s - m) - log_total for s in scores]