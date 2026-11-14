"""Finite saved-FEM diagnostic magnetic coupling; durable per-call ledger."""
from pathlib import Path
from functools import lru_cache
import hashlib
import io
import json
import os
import sqlite3
import time
from shell_core import save,exclusive,now,sha

def connect(root):
 db=sqlite3.connect(str(Path(root)/'field_calls.sqlite'),timeout=60)
 db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA busy_timeout=60000')
 return db

def initialize(root):
 with connect(root) as db:
  db.execute('CREATE TABLE calls (id TEXT PRIMARY KEY, status TEXT NOT NULL, input_sha TEXT NOT NULL, order_n INTEGER NOT NULL, started TEXT NOT NULL, completed TEXT, payload BLOB, payload_sha TEXT, error TEXT)')

def read_field(root,key):
 import numpy as np
 with connect(root) as db:row=db.execute('SELECT status,payload,payload_sha FROM calls WHERE id=?',(key,)).fetchone()
 if row is None or row[0]!='committed':raise RuntimeError('No complete field checkpoint: '+key)
 if hashlib.sha256(row[1]).hexdigest()!=row[2]:raise RuntimeError('Field BLOB digest mismatch')
 array=np.load(io.BytesIO(row[1]),allow_pickle=False)
 if array.shape!=(155,74,3) or not np.isfinite(array).all():raise RuntimeError('Field BLOB shape/finiteness mismatch')
 return array

@lru_cache(maxsize=1)
def inputs(root_text):
 import numpy as np
 root=Path(root_text);identity=json.loads((root/'input_identity.json').read_text())
 if sha(root/'inputs.npz')!=identity['inputs_npz_sha256']:raise RuntimeError('Frozen scientific inputs changed')
 with np.load(root/'inputs.npz',allow_pickle=False) as data:result={k:data[k] for k in data.files}
 if sha(root/'inputs.npz')!=identity['inputs_npz_sha256']:raise RuntimeError('Input archive changed while reading')
 return result

@lru_cache(maxsize=1)
def baseline96(root_text):return read_field(root_text,'baseline_o96')

def field_once(root_text,key,centers,frames,order):
 import numpy as np
 from magnetic_core import contributions
 root=Path(root_text);data=inputs(root_text)
 if centers.shape!=(155,3) or frames.shape!=(155,3,3) or not np.isfinite(centers).all() or not np.isfinite(frames).all():raise RuntimeError('Field input contract mismatch')
 h=hashlib.sha256((sha(root/'source_manifest.json')+str(order)).encode())
 for a in (centers,frames,data['sensors_mm']):h.update(np.asarray(a,dtype=np.float64).tobytes())
 signature=h.hexdigest()
 with connect(root) as db:
  existing=db.execute('SELECT status,input_sha FROM calls WHERE id=?',(key,)).fetchone()
  if existing:
   if existing!=('committed',signature):raise RuntimeError('Unknown/failed or changed call cannot be retried: '+key)
   return read_field(root_text,key)
  if db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]>=6:raise RuntimeError('Frozen field-call dose exceeded')
  db.execute('INSERT INTO calls(id,status,input_sha,order_n,started) VALUES(?,?,?,?,?)',(key,'started',signature,order,now()))
 try:
  values=contributions(centers,frames,data['sensors_mm'],order=order)
  if values.shape!=(155,74,3) or not np.isfinite(values).all():raise RuntimeError('Nonfinite magnetic field')
  buffer=io.BytesIO();np.save(buffer,values,allow_pickle=False);payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
  with connect(root) as db:
   db.execute('UPDATE calls SET status=?,completed=?,payload=?,payload_sha=? WHERE id=? AND status=?',('committed',now(),payload,digest,key,'started'))
  return values
 except BaseException as exc:
  with connect(root) as db:db.execute('UPDATE calls SET status=?,error=? WHERE id=? AND status=?',('failed',repr(exc),key,'started'))
  raise

def compute_baseline(root_text,cpu):
 os.sched_setaffinity(0,{cpu});start,ct=time.perf_counter(),time.process_time();data=inputs(root_text)
 field_once(root_text,'baseline_o96',data['centers_mm'],data['frames'],96)
 return {'cpu_seconds':time.process_time()-ct,'wall_seconds':time.perf_counter()-start,'field_calls':1}

def hall_delta(data,world,baseline):
 import numpy as np
 total=np.einsum('pm,msi->psi',data['polarities'],world-baseline)
 return np.einsum('sij,psi->psj',data['sensor_frames'],total)

def rms(value):
 import numpy as np
 return float(np.sqrt(np.mean(np.asarray(value)**2)))

def compare_vectors(test,reference,indices):
 import numpy as np
 test=np.asarray(test)[indices];reference=np.asarray(reference)[indices];difference=test-reference
 reference_rms=rms(reference);absolute=rms(difference)
 node_ref=np.sqrt(np.mean(reference**2,axis=1));node_error=np.sqrt(np.mean(difference**2,axis=1))
 relative=np.divide(node_error,node_ref,out=np.full_like(node_ref,np.nan),where=node_ref!=0.)
 finite=np.isfinite(relative);worst=int(np.nanargmax(relative)) if finite.any() else None;weak=int(np.argmin(node_ref))
 return {'absolute_RMS_uT_per_T':absolute,'reference_RMS_uT_per_T':reference_rms,
  'relative_RMS':absolute/reference_rms if reference_rms else None,
  'zero_reference_nodes':int(np.count_nonzero(node_ref==0.)),'nonfinite_relative_nodes':int((~finite).sum()),
  'maximum_finite_node_relative':float(relative[worst]) if worst is not None else None,
  'worst_node_global_index':int(indices[worst]) if worst is not None else None,
  'worst_node_same_reference_RMS':float(node_ref[worst]) if worst is not None else None,
  'worst_node_same_absolute_error_RMS':float(node_error[worst]) if worst is not None else None,
  'weakest_node_global_index':int(indices[weak]),'weakest_node_reference_RMS':float(node_ref[weak]),
  'weakest_node_same_absolute_error_RMS':float(node_error[weak])}

def run_condition(root_text,spec,cpu):
 import numpy as np
 root=Path(root_text);plan=json.loads((root/'simulation_plan.json').read_text());os.sched_setaffinity(0,{cpu})
 name=f"h{spec['h']:g}_nu{spec['nu']:g}_{spec['load']}";out=root/'conditions'/name;out.mkdir(parents=True,exist_ok=True)
 exclusive(out/'started.json',{'time':now(),'spec':spec,'pid':os.getpid(),'cpu':cpu})
 start,ct=time.perf_counter(),time.process_time();data=inputs(root_text)
 parent=Path(plan['motion_roots'][str(int(spec['h']))]);path=parent/f"blocks/h{spec['h']:g}/nu{spec['nu']:g}/{spec['load']}/source_motion.npz"
 digest=sha(path)
 with np.load(path,allow_pickle=False) as saved:motion={k:saved[k] for k in ('source_centers_mm','nominal_frames','moved_centers_mm','moved_frames')}
 if sha(path)!=digest:raise RuntimeError('Saved source motion changed')
 if not np.array_equal(motion['source_centers_mm'],data['centers_mm']) or not np.array_equal(motion['nominal_frames'],data['frames']):raise RuntimeError('Source geometry/order/frame mismatch')
 responses={};quadrature=[];response_metrics=[]
 for mode in plan['motion_modes']:
  frames=data['frames'] if mode=='translation_only' else motion['moved_frames']
  deltas=[]
  for order in (48,96):
   key=name+'_'+mode+f'_o{order}';world=field_once(root_text,key,motion['moved_centers_mm'],frames,order)
   base=data['baseline48'] if order==48 else baseline96(root_text)
   delta=hall_delta(data,world,base);responses[f'{mode}_o{order}']=delta;deltas.append(delta)
  denominator=plan['quadrature']['delta_absolute_uT_per_T']+plan['quadrature']['delta_relative']*np.maximum(np.abs(deltas[0]),np.abs(deltas[1]))
  ratio=float(np.max(np.abs(deltas[1]-deltas[0])/denominator));quadrature.append({'mode':mode,'max_gate_ratio':ratio,'passed':ratio<=1.})
  for pi,polarity in enumerate(plan['polarity_names']):
   for layout,indices in plan['layouts'].items():
    delta=deltas[1][pi,indices];node=np.sqrt(np.mean(delta**2,axis=1));weak=int(np.argmin(node));strong=int(np.argmax(node))
    response_metrics.append({'mode':mode,'polarity':polarity,'layout':layout,'RMS_uT_per_T':rms(delta),
     'weakest_node_RMS_uT_per_T':float(node[weak]),'weakest_node_global_index':int(indices[weak]),
     'strongest_node_RMS_uT_per_T':float(node[strong]),'strongest_node_global_index':int(indices[strong]),
     'zero_node_response_count':int((node==0.).sum()),'nonfinite_values':int((~np.isfinite(delta)).sum())})
 omission=[]
 for pi,polarity in enumerate(plan['polarity_names']):
  for layout,indices in plan['layouts'].items():
   omission.append({'polarity':polarity,'layout':layout,**compare_vectors(responses['translation_only_o96'][pi],responses['translation_and_rotation_o96'][pi],indices)})
 np.savez_compressed(out/'responses.npz',**responses)
 passed=all(q['passed'] for q in quadrature)
 result={'spec':spec,'input_motion_npz_sha256':digest,'quadrature':quadrature,'passed_quadrature':passed,
  'responses':response_metrics,'rotation_omission':omission,'field_calls':4,
  'cpu_seconds':time.process_time()-ct,'wall_seconds':time.perf_counter()-start,'cpu':cpu}
 save(out/'result.json',result)
 if not passed:raise RuntimeError('Delta-field quadrature gate failed: '+name)
 return result
