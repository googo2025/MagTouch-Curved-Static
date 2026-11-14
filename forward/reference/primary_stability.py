"""Evaluate the original numerical stability rule for the paper's primary architecture."""

from paper1_batch_protocol import batch_scores

PRIMARY = (51, 64, 64, 3)


def check_primary(batches):
    recent = batches[-3:]
    if len(recent) != 3:
        raise ValueError("Three consecutive training cohorts are required")
    scores = [batch_scores(records)[PRIMARY] for records in recent]
    stable_seeds = [sum(record["stable"] for record in records
                        if tuple(record["architecture"]) == PRIMARY)
                    for records in recent]
    improvements = [a-b for a,b in zip(scores, scores[1:])]
    limits = [max(.05, .01*a) for a in scores[:-1]]
    passed = stable_seeds == [3,3,3] and all(
        0 <= change < limit for change,limit in zip(improvements, limits))
    return dict(passed=passed, scores_mm=scores, stable_seeds=stable_seeds,
                improvement_mm=improvements, limits_mm=limits)
