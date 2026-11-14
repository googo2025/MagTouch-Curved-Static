"""Build nested training cohorts and test the frozen learning-curve gate."""

from collections import deque
from statistics import mean

WIDTHS = {(51, 16, 3), (51, 64, 64, 3)}
SEEDS = {20261110, 20261111, 20261112}


def region(row):
    z = row['xyz'][2]
    return 'side' if z < 95 else 'seam' if z <= 105 else 'cap'


def subsequent_cohorts(first_train, accepted_train, batch_size=24):
    known = {row['load']: row for row in accepted_train}
    if len(known) != len(accepted_train):
        raise ValueError('Duplicate accepted training IDs')
    used = {row['load'] for row in first_train}
    if len(used) != len(first_train) or any(known.get(row['load']) != row for row in first_train):
        raise ValueError('Frozen cohort is inconsistent with accepted training ledger')
    queues = {label: deque() for label in ('side', 'seam', 'cap')}
    for row in accepted_train:
        if row['load'] not in used:
            queues[region(row)].append(row)
    cohort = list(first_train)
    while any(queues.values()):
        added = []
        while len(added) < batch_size and any(queues.values()):
            for queue in queues.values():
                if queue and len(added) < batch_size:
                    added.append(queue.popleft())
        cohort.extend(added)
        yield list(cohort), len(added)


def batch_scores(records):
    scores = {widths: [] for widths in WIDTHS}
    seen = set()
    for record in records:
        widths = tuple(record['architecture'])
        seed = record['seed']
        if widths not in WIDTHS or seed not in SEEDS or (widths, seed) in seen:
            raise ValueError('Unexpected or duplicate architecture/seed')
        seen.add((widths, seed))
        scores[widths].append(record['best']['mean'])
    if len(seen) != len(WIDTHS) * len(SEEDS):
        raise ValueError('Incomplete paired batch')
    return {widths: mean(values) for widths, values in scores.items()}


def dataset_stable(batches):
    """Require two consecutive small improvements for each architecture."""
    recent = batches[-3:]
    if len(recent) < 3:
        return False
    scores = [batch_scores(batch) for batch in recent]
    if not all(r['stable'] for batch in recent for r in batch):
        return False
    return all(0 <= a[widths] - b[widths] < max(.05, .01 * a[widths])
               for a, b in zip(scores, scores[1:]) for widths in WIDTHS)
