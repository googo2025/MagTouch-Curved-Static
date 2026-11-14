import bisect,datetime,hashlib,json,math,os,socket
from pathlib import Path
assert socket.gethostname()=='REFERENCE_HOST' and Path('/etc/machine-id').read_text().strip()=='REFERENCE_MACHINE_ID'
base=Path('/path/to/reference-study')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def alive(i):
    try:return Path(f"/proc/{i['pid']}/stat").read_text().rsplit(')',1)[1].split()[19]==i['proc_start_ticks']
    except (FileNotFoundError,ProcessLookupError):return False
def verify(root,digest):
    m=root/'source_manifest.json'
    assert sha(m)==digest and all(Path(n).name==n and sha(root/n)==h for n,h in read(m)['files'].items())
def collect_rows(include_test_gap=False):
    parent=base/'sim436_basic_grid_regional_reeval'
    verify(parent,'2bb3c16785940b48fbf870d1f7158723831a2f244cf6407bf62697fa9cbdb52e')
    accepted=read(parent/'aggregate_report.json')
    split_path=parent/'qualified_dataset_split.json'
    assert accepted['passed'] and accepted['dataset_sha256']==sha(split_path)
    split=read(split_path)
    p={'layout':'original17','polarity':'checker'}
    new_rows=[]
    sources = (
        ('S438','sim438_interpolation_checks',3132116,'163740714','506a912d3263d0d7c8798ece4c885c7d0f13a00e8afa75bc1336a4537a0849ef'),
        ('S439','sim439_interpolation_followup',3134002,'163760356','9d1cf35f60446ab2bf526b0a9fff0dc64f0bb5840976f72971b3addd0a23b82c'))

    if include_test_gap:
        sources += (('paper1_gap','paper1_test_gap_20261110',3190911,'164403577',
                     '430060f3d5709c4a59008e1b2db1d3814c185c1d15559ca757fbc77459975bd7'),)
    outside_anchors = []
    for key,folder,pid,tick,digest in sources:
        q=base/folder; verify(q,digest)
        i=read(q/'execution_identity.json')
        assert (i['pid'],i['proc_start_ticks'],i['source_manifest_sha256'])==(pid,tick,digest)
        state=read(q/'state.json')
        if include_test_gap and (alive(i) or state['stage'] != 'expanded_wave_complete' or state.get('error')):
            raise RuntimeError(f'Queue has not closed cleanly: {key}')
        first = len(new_rows)
        for run in state['runs']:
            if '-MAGNETIC-' not in run['job_id'] or alive(run):continue
            m=Path(run['remote_root']);report=m/'aggregate_report.json'
            if not report.is_file():continue
            verify(m,run['source_manifest_sha256'])
            identity=read(m/'execution_identity.json')
            assert all(identity[k]==run[k] for k in ('job_id','pid','proc_start_ticks','source_manifest_sha256'))
            mr=read(report)
            assert mr['job_id']==run['job_id'] and mr['passed'] and mr['cap_load_change_order_all_resolved']
            assert mr['case_ledger_counts']['inner2.5']==[['imported','committed',2],['new','committed',4]]
            force=read(m/'simulation_plan.json')['traction_parent'];f=Path(force['root'])
            verify(f,force['source_manifest_sha256'])
            assert not alive(force)
            fr=read(f/'aggregate_report.json')
            assert fr['passed'] and fr['conditions']==2 and all(x['passed'] and all(x['gates'].values()) for x in fr['results'])
            for load in read(f/'simulation_plan.json')['loads']:
                assert load['Gaussian_sigma_mm']==12
                file=m/'inner2.5'/'conditions'/('nu0.49_'+load['id'])/'responses.npz'
                new_rows.append(dict(folder=m.name,load=load['id'],xyz=load['center_mm'],sha=sha(file)))
        if key == 'S438':
            outside_anchors = list(new_rows[first:])
        if key == 'paper1_gap':
            assert len(new_rows) - first == 18
    assert len({x['load'] for x in split['rows']+new_rows})==len(split['rows'])+len(new_rows)
    from collections import Counter
    def role(xyz):
        x,y,z=xyz
        phi=math.degrees(math.atan2(y,x))%360
        beta=math.degrees(math.acos(max(-1,min(1,(z-100)/20)))) if z>100 else None
        for name,lo,hi,level,a,b in (
            ('test_side',70,110,z,0,35),('dev_side',115,155,z,0,35),
            ('test_seam',300,330,z,95,105),('dev_seam',250,280,z,95,105),
            ('test_cap',205,245,beta,40,70),('dev_cap',115,155,beta,40,70)):
            if lo-1e-8<=phi<=hi+1e-8 and level is not None and a-1e-8<=level<=b+1e-8:
                return name+('_side' if z<=100 else '_cap') if 'seam' in name else name
        return 'train'
    legacy=set(split['legacy'])|set(split['region'])
    assert len(legacy)==17
    assert len(split['rows']) == 150
    historical = [row for row in split['rows'] if row['load'] in legacy]
    assert len(historical) == 17
    seen = []
    groups = dict(train=[], dev=[], final=[], historical=historical)
    for original, rows in ((True, split['rows']), (False, new_rows)):
        for row in rows:
            xyz = row['xyz']
            assert len(xyz) == 3 and all(math.isfinite(x) for x in xyz)
            if any(math.dist(xyz, prev) <= 1e-7 for prev in seen):
                raise ValueError("Duplicate physical coordinate in accepted ledger")
            seen.append(xyz)
            if row['load'] in legacy:
                assert original
                continue
            label = role(xyz)
            tagged = dict(row, region=label)
            if label == 'train':
                groups['train'].append(tagged)
            elif label.startswith('dev_'):
                groups['dev'].append(tagged)
            elif not original:
                groups['final'].append(tagged)
        if original:
            baseline_count = len(groups['train'])
    if include_test_gap:
        counts = Counter(row['region'] for row in groups['final'])
        quotas = dict(test_side=10, test_seam_side=5, test_seam_cap=5, test_cap=10)
        if any(counts[k] < n for k,n in quotas.items()):
            raise RuntimeError('Independent evaluation quota unmet')
        grid_rows = split['rows']
        anchor_rows = [row for row in grid_rows + outside_anchors
                       if row['load'] not in legacy and role(row['xyz']) == 'train']
        assert len(anchor_rows) == 147
        return groups, baseline_count, grid_rows, anchor_rows, role
    return groups, baseline_count


def load_features(rows):
    import numpy as np

    model = read(base/'sim289_cap_azimuth337p5_magnetic'/'simulation_plan.json')
    indices = model['layouts']['original17']
    polarity = model['polarity_names'].index('checker')
    assert len(indices) == len(set(indices)) == 17
    vectors = []
    for row in rows:
        path = base/row['folder']/'inner2.5'/'conditions'/('nu0.49_'+row['load'])/'responses.npz'
        assert sha(path) == row['sha'] and read(path.parent/'result.json')['passed']
        with np.load(path, allow_pickle=False) as archive:
            field = archive['order96']
            assert field.shape == (3,74,3)
            vector = field[polarity, indices, :].ravel()
            assert np.isfinite(vector).all() and np.linalg.norm(vector) > 0
            vectors.append(vector)
    return np.asarray(vectors), np.asarray([r['xyz'] for r in rows])

