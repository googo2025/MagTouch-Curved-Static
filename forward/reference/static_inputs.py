import hashlib,json,socket
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'sim436_basic_grid_regional_reeval'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_inputs():
    assert socket.gethostname()=='REFERENCE_HOST' and Path('/etc/machine-id').read_text().strip()=='REFERENCE_MACHINE_ID'
    manifest=SOURCE/'source_manifest.json'
    assert sha(manifest)=='2bb3c16785940b48fbf870d1f7158723831a2f244cf6407bf62697fa9cbdb52e'
    assert all(sha(SOURCE/n)==h for n,h in read(manifest)['files'].items())
    report=read(SOURCE/'aggregate_report.json')
    split_path=SOURCE/'qualified_dataset_split.json'
    assert report['passed'] and sha(split_path)==report['dataset_sha256']
    assert sha(SOURCE/'mlp_weights.npz')==report['weights_sha256']
    split=read(split_path)
    reference=read(BASE/'sim289_cap_azimuth337p5_magnetic'/'simulation_plan.json')
    indices=reference['layouts']['original17']
    polarity=reference['polarity_names'].index('checker')
    rows=split['rows']
    vectors=[]
    for row in rows:
        p=BASE/row['folder']/'inner2.5'/'conditions'/('nu0.49_'+row['load'])/'responses.npz'
        assert sha(p)==row['sha'] and read(p.parent/'result.json')['passed']
        with np.load(p,allow_pickle=False) as z:
            a=z['order96']
            assert a.shape==(3,74,3) and np.isfinite(a).all()
            vectors.append(a[polarity,indices,:].ravel())
    held=set(split['legacy']) | set(split['region'])
    test=np.array([r['load'] in held for r in rows])
    assert len(rows)==150 and len(held)==test.sum()==17
    assert held=={x['site'] for x in report['comparisons']}
    return np.array(vectors),np.array([r['xyz'] for r in rows]),test,rows,report

def project(q):
    x,y,z=q
    angle=np.arctan2(y,x)
    side=np.array([20*np.cos(angle),20*np.sin(angle),np.clip(z,0,100)])
    d=np.array([x,y,max(z-100,0)])
    if np.linalg.norm(d)==0:
        return side
    cap=np.array([0,0,100])+20*d/np.linalg.norm(d)
    return min((side,cap),key=lambda p:np.linalg.norm(p-q))

def baseline(x,y,test):
    with np.load(SOURCE/'mlp_weights.npz',allow_pickle=False) as z:
        center,scale,w,b,v,c=[z[k] for k in ('center','scale','w','b','v','c')]
    bank=x[~test]
    bank=bank/np.linalg.norm(bank,axis=1,keepdims=True)
    labels=y[~test]
    def predict(vector):
        assert vector.shape==(51,) and np.isfinite(vector).all()
        norm=np.linalg.norm(vector)
        if norm<=0:raise ValueError('zero field')
        raw=(np.tanh(((vector-center)/scale)@w+b)@v+c)*[20,20,60]+[0,0,60]
        return project(raw),labels[np.argmax(bank@(vector/norm))]
    return predict
