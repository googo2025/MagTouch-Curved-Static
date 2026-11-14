"""Run the sealed spatial evaluation after both authorized predecessors close."""

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from paper1_evaluation_inputs import alive, base, collect_rows, load_features, read, sha, verify
from primary_stability import check_primary


def require_closed(folder, expected):
    root = base / folder
    identity = read(root / 'execution_identity.json')
    assert (identity['pid'], identity['proc_start_ticks'], identity['source_manifest_sha256']) == expected
    verify(root, expected[2])
    while alive(identity):
        time.sleep(15)
    if (root/'failure.json').exists():
        raise RuntimeError(f'Predecessor failed: {folder}')
    return root


def require_closed_result_commit(gate_receipt):
    previous = base/'paper1_primary_820_evaluation_20261113'
    digest = '58a319aeb2e48e0ca1f89414bfc0d365098ea53037482c5592cf0155da8d3f9e'
    verify(previous,digest)
    identity = read(previous/'execution_identity.json')
    assert (identity['pid'],identity['proc_start_ticks'],identity['source_manifest_sha256']) == (2314497,'11591933',digest)
    assert not alive(identity) and not (previous/'aggregate_report.json').exists()
    assert read(previous/'failure.json')['error'] == "TypeError('Object of type bool is not JSON serializable')"
    assert read(previous/'primary_freeze_receipt.json') == gate_receipt
    return previous


def main():
    import shutil

    plan = read(ROOT/'simulation_plan.json')
    assert str(ROOT) == plan['remote_root']
    verify(ROOT, sha(ROOT/'source_manifest.json'))
    assert not (ROOT/'aggregate_report.json').exists()
    with (ROOT/'execution_identity.json').open('x') as receipt:
        json.dump(dict(job_id=plan['job_id'], pid=os.getpid(),
                       proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                       source_manifest_sha256=sha(ROOT/'source_manifest.json')), receipt, ensure_ascii=True)
    earlier = require_closed('paper1_static_boundary_extend48_20261112', (
        1247149,'1831775','912080b09943ae3427867da2156caae12adec8f7ee9cc5ea293fdd93a4780490'))
    later = require_closed('paper1_static_boundary_next48_20261112', (
        1964920,'2922707','fee3fe7bc39d627c4e51674fc3edeac15942aed18f16e20085621520c7cf10ac'))
    gap = require_closed('paper1_test_gap_20261110', (
        3190911,'164403577','430060f3d5709c4a59008e1b2db1d3814c185c1d15559ca757fbc77459975bd7'))
    assert read(gap/'state.json')['stage'] == 'expanded_wave_complete'
    former = read(earlier/'aggregate_report.json')
    latter = read(later/'aggregate_report.json')
    assert latter['first_report_sha256'] == sha(earlier/'aggregate_report.json')
    entries = [(earlier,former['batches'][-1])] + [(later,item) for item in latter['batches']]
    assert [item['train_sites'] for _,item in entries] == [772,796,820]
    assert [item['batch'] for _,item in entries] == [27,28,29]
    assert all(sha(earlier/n) == sha(later/n) == sha(ROOT/n)
               for n in ('paper1_static_fit.py','paper1_batch_protocol.py','static_inputs.py'))
    reports, splits = [], []
    for folder,entry in entries:
        model_dir = folder/f"batch_{entry['batch']:02d}"
        report = read(model_dir/'report.json')
        split = read(model_dir/'split.json')
        assert report['dataset_sha256'] == entry['dataset_sha256'] == sha(model_dir/'split.json')
        assert report['train_sites'] == entry['train_sites'] == len(split['train'])
        assert report['dev_sites'] == len(split['dev']) == 17
        assert len(split['historical']) == 17
        assert all(m['history'][-1]['step'] == 12000 for m in report['models'])
        reports.append(report['models'])
        splits.append(split)
    for a,b in zip(splits,splits[1:]):
        assert b['train'][:len(a['train'])] == a['train']
        assert b['dev'] == a['dev'] and b['historical'] == a['historical']
    evidence = check_primary(reports)
    assert evidence['passed']
    evidence.update(train_sites=[772,796,820],cohort_report_sha256=sha(model_dir/'report.json'))
    checked = [evidence]
    final_batch = entries[-1][1]
    model_report = read(model_dir/'report.json')
    split_file = model_dir/'split.json'
    assert model_report['dataset_sha256'] == final_batch['dataset_sha256'] == sha(split_file)
    frozen = read(split_file)
    weights_sha256 = {}
    for record in model_report['models']:
        widths = record['architecture']
        label = f"static_{len(widths)-2}_{record['seed']}"
        weights = model_dir/(label+'.npz')
        assert sha(weights) == record['weights_sha256'] and record['history'][-1]['step'] == 12000
        weights_sha256[label] = record['weights_sha256']
    gate_receipt = dict(rule=plan['primary_gate'], selected=checked[-1], checks=checked,
                        dataset_sha256=sha(split_file), model_report_sha256=sha(model_dir/'report.json'),
                        weights_sha256=weights_sha256)
    failed = require_closed_result_commit(gate_receipt)
    assert sha(failed/'primary_freeze_receipt.json')
    with (ROOT/'primary_freeze_receipt.json').open('x') as receipt:
        json.dump(gate_receipt, receipt, ensure_ascii=True, allow_nan=False)
        receipt.flush()
        os.fsync(receipt.fileno())
    available = next(int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()
                     if row.startswith('MemAvailable:'))
    assert available >= 8 * 1024**2 + 256 * 1024
    assert shutil.disk_usage(ROOT).free >= 20 * 1024**3
    os.sched_setaffinity(0, {4})
    groups, _, base_rows, anchor_rows, role = collect_rows(include_test_gap=True)
    train_ids = {r['load'] for r in frozen['train']}
    test_ids = {r['load'] for r in groups['final']}
    assert final_batch['train_sites'] == len(frozen['train']) == len(train_ids) == 820
    assert all(role(row['xyz']) == 'train' for row in frozen['train'])
    assert all(math.dist(a['xyz'], b['xyz']) > 1e-7
               for a in frozen['train'] for b in groups['final']+groups['dev']+groups['historical'])
    assert frozen['dev'] == groups['dev']
    assert frozen['historical'] == [r['load'] for r in groups['historical']]
    assert not test_ids & (train_ids | {r['load'] for r in frozen['dev']} | set(frozen['historical']))

    from collections import Counter
    assert Counter(r['region'] for r in groups['final']) == {
        'test_side':10, 'test_seam_side':5, 'test_seam_cap':5, 'test_cap':10}
    assert len(groups['historical']) == 17

    from paper1_batch_protocol import batch_scores
    from paper1_evaluate_pairs import engineering_grid, evaluate, load_predictor
    from paper1_paired_metrics import position_errors
    from static_inputs import project

    batch_scores(model_report['models'])
    predictors = {}
    for record in model_report['models']:
        widths = record['architecture']
        key = f"static_{len(widths)-2}_{record['seed']}"
        weights = model_dir/(key+'.npz')
        assert (widths != [51,64,64,3] or record['stable']) and sha(weights) == weights_sha256[key]
        predictors[key] = load_predictor(weights, widths, project)

    import numpy as np
    train_fields, train_xyz = load_features(frozen['train'])
    normalized = train_fields / np.linalg.norm(train_fields, axis=1, keepdims=True)
    predictors['template'] = lambda field: train_xyz[np.argmax(normalized @ (field/np.linalg.norm(field)))]
    grid, allowed = engineering_grid(base_rows, load_features, role,
        set(frozen['historical']) | {r['load'] for r in groups['dev']} | test_ids)
    anchor_fields, anchor_xyz = load_features(anchor_rows)
    direct, _ = load_features(groups['final'])
    result = evaluate(groups['final'], direct, grid, allowed, anchor_xyz, anchor_fields, predictors)
    historic, historic_xyz = load_features(groups['historical'])
    result['historical_direct'] = [
        dict(site=row['load'], model=key, **position_errors(xyz, predict(field)))
        for row, xyz, field in zip(groups['historical'], historic_xyz, historic)
        for key,predict in predictors.items()
    ]
    result['recovered_from_freeze_sha256'] = sha(failed/'primary_freeze_receipt.json')
    result['primary_freeze_sha256'] = sha(ROOT/'primary_freeze_receipt.json')
    result['dataset_sha256'] = sha(split_file)
    result['model_report_sha256'] = sha(model_dir/'report.json')
    target = ROOT/'aggregate_report.json'
    assert not target.exists()
    pending = ROOT/'aggregate_report.partial'
    with pending.open('x') as stream:
        json.dump(result, stream, ensure_ascii=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(target)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if (ROOT/'execution_identity.json').exists():
            with (ROOT/'failure.json').open('x') as stream:
                json.dump(dict(error=repr(exc)), stream, ensure_ascii=True)
        raise



