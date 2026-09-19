import math

def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]  # all values in (0, 1]
    total = sum(exps)
    return [e / total for e in exps]