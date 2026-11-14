"""Exact parent-gated continuous one-way mechanics-to-field successor."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from shell_core import save,exclusive,now,sha


def verify_parent(parent,wait=False):
 root=Path(parent['root'])
 while True:
  if sha(root/'source_manifest.json')!=parent['source_manifest_sha256']:raise RuntimeError('Parent manifest identity mismatch')
  for name,pin in json.loads((root/'source_manifest.json').read_text())['files'].items():
   if Path(name).name!=name or sha(root/name)!=pin:raise RuntimeError('Parent source mismatch')
  ident=json.loads((root/'execution_identity.json').read_text())
  if any(ident[k]!=parent[k] for k in ('job_id','pid','proc_start_ticks','source_manifest_sha256')):raise RuntimeError('Parent exact process identity mismatch')
  proc=Path(f"/proc/{ident['pid']}/stat");alive=proc.exists() and proc.read_text().rsplit(')',1)[1].split()[19]==ident['proc_start_ticks']
  state=json.loads((root/'state.json').read_text())
  if state['stage']=='failed_preserved':raise RuntimeError('Preserved parent failure; no magnetic submission')
  if not alive:
   if state['stage']!='finite_stage_complete' or not json.loads((root/'aggregate_report.json').read_text())['passed']:raise RuntimeError('Parent did not pass')
   return root
  if not wait:raise RuntimeError('Non-waiting input parent is active')
  save(ROOT/'state.json',{'stage':'waiting_exact_two_qualified_side_tractions','observed_at':now(),'new_field_calls':0,'scientific_workers':0})
  import psutil
  try:
   budget_root=psutil.Process(ident['pid']);rss=psutil.Process().memory_info().rss
   for pr in [budget_root]+budget_root.children(recursive=True):
    try:rss+=pr.memory_info().rss
    except psutil.NoSuchProcess:pass
   seen={os.getpid(),budget_root.pid}|{pr.pid for pr in budget_root.children(recursive=True)}
   plan=json.loads((ROOT/'simulation_plan.json').read_text())
   for item in plan['upstream_wait_budget']:
    st=Path(f"/proc/{item['pid']}/stat")
    if st.exists() and st.read_text().rsplit(')',1)[1].split()[19]==item['proc_start_ticks']:
     try:
      up=psutil.Process(item['pid'])
      for pr in [up]+up.children(recursive=True):
       if pr.pid not in seen:
        seen.add(pr.pid);rss+=pr.memory_info().rss
     except psutil.NoSuchProcess:pass
   if rss>8*2**30 or psutil.virtual_memory().available<8*2**30 or psutil.disk_usage(str(ROOT)).free<20*2**30:raise RuntimeError('Combined own parent/waiter resource guard')
  except psutil.NoSuchProcess:
   continue  # Reverify exact final parent state after exit race, no fake RSS.
  time.sleep(5)


def condition(setting,cpu):
 import numpy as np
 from coupling_core import inputs,field_once,read_field,hall_delta,rms,compare_vectors
 master=Path(__file__).resolve().parent;p=json.loads((master/'simulation_plan.json').read_text());ROOT=master/setting['case'];os.sched_setaffinity(0,{cpu});data=inputs(str(ROOT))
 name=f"nu{setting['nu']:g}_{setting['load']}";out=ROOT/'conditions'/name;out.mkdir(parents=True,exist_ok=True)
 exclusive(out/'started.json',{'at':now(),'setting':setting,'pid':os.getpid(),'cpu':cpu})
 start,ct=time.perf_counter(),time.process_time();folder=Path(p['traction_parent']['root'])/setting['case']/name
 report=json.loads((folder/'result.json').read_text())
 if not report['passed'] or not all(report['gates'].values()):raise RuntimeError('Individual mechanical condition not accepted')
 posepath=folder/'rigid_source_motion.npz';pin=sha(posepath)
 with np.load(posepath,allow_pickle=False) as pose:
  if not np.array_equal(pose['source_centers_mm'],data['centers_mm']) or not np.array_equal(pose['nominal_frames'],data['frames']):raise RuntimeError('Rigid pose geometry/frame/order mismatch')
  centers=pose['moved_centers_mm'];frames=pose['moved_frames']
 if sha(posepath)!=pin:raise RuntimeError('Mechanical pose changed during read')
 relative=np.einsum('msi,mij->msj',data['sensors_mm'][None]-centers[:,None],frames)
 inside=(np.linalg.norm(relative[:,:,:2],axis=2)<=2.5)&(np.abs(relative[:,:,2])<=.5)
 if np.any(inside):raise RuntimeError('Base-fixed Hall point inside a moved finite cylinder; no field submission')
 save(out/'observer_domain_check.json',{'inside_moved_cylinder_count':int(inside.sum()),'base_fixed_Hall_carrier_assumption':True})
 responses={}
 for order in (48,96):
  world=field_once(str(ROOT),name+f'_o{order}',centers,frames,order)
  responses[f'order{order}']=hall_delta(data,world,read_field(str(ROOT),f'baseline_o{order}'))
 a,b=responses['order48'],responses['order96'];allowed=1e-8+1e-4*np.maximum(np.abs(a),np.abs(b));ratio=np.abs(a-b)/allowed
 worst=np.unravel_index(np.argmax(ratio),ratio.shape);metrics=[]
 per=world-read_field(str(ROOT),'baseline_o96')
 for pi,polarity in enumerate(p['polarity_names']):
  for layout,indices in p['layouts'].items():
   value=b[pi,indices];node=np.sqrt(np.mean(value**2,axis=1));weak=int(np.argmin(node));si=indices[weak]
   sources=data['polarities'][pi,:,None]*per[:,si];den=np.linalg.norm(sources.sum(axis=0));num=np.linalg.norm(sources,axis=1).sum()
   metrics.append({'polarity':polarity,'layout':layout,'RMS_uT_per_T':rms(value),'RMS_per_axis_N_uT_per_TN':rms(value)/.01,
    'weakest_global_index':int(si),'weakest_coordinate_mm':data['sensors_mm'][si].tolist(),'weakest_node_RMS_uT_per_T':float(node[weak]),
    'weakest_source_cancellation_factor':float(num/den) if den else None,'maximum_node_RMS_uT_per_T':float(node.max()),
    'zero_nodes':int((node==0).sum()),'nonfinite_values':int((~np.isfinite(value)).sum())})
 ref=p['meridian_references'][setting['load']]
 oldroot=Path(ref['magnetic']['root'])/setting['case'];oldname='nu0.49_'+ref['load_id'];oldpath=oldroot/'conditions'/oldname/'responses.npz';oldpin=sha(oldpath)
 if str(oldpath)!=ref['responses'][setting['case']] or oldpin!=p['saved_reference_input_pins'][str(oldpath)]:raise RuntimeError('Exact archived phi225 reference path and SHA')
 if sha(oldroot/'inputs.npz')!=sha(ROOT/'inputs.npz'):raise RuntimeError('Same-design Hall/source input mismatch')
 oldreport=json.loads((oldpath.parent/'result.json').read_text())
 if not oldreport['passed']:raise RuntimeError('Unaccepted same-design oblique cap response')
 with np.load(oldpath,allow_pickle=False) as z:old48=z['order48'];old96=z['order96']
 if sha(oldpath)!=oldpin or old48.shape!=a.shape or old96.shape!=b.shape:raise RuntimeError('Saved oblique cap response changed')
 angular48=a-old48;angular96=b-old96
 cratio=np.abs(angular48-angular96)/(1e-10+1e-3*np.maximum(np.abs(angular48),np.abs(angular96)))
 comparisons=[]
 for pi,polarity in enumerate(p['polarity_names']):
  for layout,indices in p['layouts'].items():
   row={'polarity':polarity,'layout':layout,**compare_vectors(b[pi],old96[pi],indices)}
   if row['worst_node_global_index'] is not None:row['worst_coordinate_mm']=data['sensors_mm'][row['worst_node_global_index']].tolist()
   row['cap_load_change_order_gate_ratio_max']=float(cratio[pi,indices].max());row['cap_load_change_order_resolved']=bool(np.all(cratio[pi,indices]<=1));comparisons.append(row)
 symmetry_diagnostics=[]  # These phi45loads do not support the cap reflection constraint.
 np.savez_compressed(out/'responses.npz',**responses)
 result={'setting':setting,'passed':bool(np.all(ratio<=1)),'maximum_delta_quadrature_gate_ratio':float(ratio[worst]),
  'worst_component_index':list(map(int,worst)),'same_absolute_quadrature_difference':float(abs(a[worst]-b[worst])),
  'same_delta96':float(b[worst]),'pose_SHA256':pin,'metrics':metrics,'old_response_SHA256':oldpin,'reference_load_id':ref['load_id'],'comparisons_to_same_design_phi225_reference':comparisons,'cap_load_difference_is_not_a_spatial_screen':True,'cap_load_change_order_resolved':bool(np.all(cratio<=1)),'cap_load_change_order_gate_ratio_max':float(cratio.max()),'new_field_calls':2,'new_FEM_calls':0,'ideal_symmetry_diagnostics':symmetry_diagnostics,
  'wall_seconds':time.perf_counter()-start,'CPU_seconds':time.process_time()-ct}
 save(out/'result.json',result)
 if not result['passed']:raise RuntimeError('Unchanged stable magnetic delta gate failed')
 return result


def main():
 import numpy as np
 import psutil
 p=json.loads((ROOT/'simulation_plan.json').read_text());start=time.perf_counter();peak=0
 if Path('/etc/machine-id').read_text().strip()!=p['machine_id'] or str(ROOT)!=p['remote_root']:raise RuntimeError('Machine/root')
 for n,h in json.loads((ROOT/'source_manifest.json').read_text())['files'].items():
  if Path(n).name!=n or sha(ROOT/n)!=h:raise RuntimeError('Own immutable source')
 exclusive(ROOT/'execution_identity.json',{'job_id':p['job_id'],'pid':os.getpid(),'proc_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'started_at':now()})
 os.sched_setaffinity(0,{7});parent=verify_parent(p['traction_parent'],True);baseline=verify_parent(p['baseline_parent']);qualified=verify_parent(p['kernel_parent'])
 a=json.loads((parent/'aggregate_report.json').read_text())
 if a['conditions']!=2 or a['new_physical_FEM_RHS']+a.get('reused_physical_FEM_RHS',0)!=2 or len(a['results'])!=2 or not all(r['passed'] and all(r['gates'].values()) for r in a['results']):raise RuntimeError('Exact two-condition physical gate')
 for name,ref in [('magnetic_core.py','stable_magnetic.py'),('legacy_core.py','legacy_core.py')]:
  if sha(ROOT/name)!=sha(qualified/ref):raise RuntimeError('Qualified byte-exact kernel')
 for ref in p['meridian_references'].values():
  verify_parent(ref['magnetic'])
  verify_parent(ref['review'])
 for path,pin in p['saved_reference_input_pins'].items():
  if sha(Path(path))!=pin:raise RuntimeError('Exact previously audited same-latitude inputs')
 from coupling_core import initialize,connect,inputs
 for label in ('inner2.5',):
  out=ROOT/label;out.mkdir(exist_ok=False);old=baseline/label;identity=json.loads((old/'input_identity.json').read_text())
  if sha(old/'inputs.npz')!=identity['inputs_npz_sha256']:raise RuntimeError('Same-design nominal SHA')
  shutil.copyfile(old/'inputs.npz',out/'inputs.npz');save(out/'input_identity.json',identity)
  # Coupling ledger policy is pinned to this subdirectory's own immutable source manifest.
  for name in ('simulation_plan.json','run_magnetic.py','coupling_core.py','magnetic_core.py','legacy_core.py','shell_core.py'):shutil.copyfile(ROOT/name,out/name)
  shutil.copyfile(ROOT/'source_manifest.json',out/'source_manifest.json')
  initialize(out)
  with connect(out) as db:db.execute("ALTER TABLE calls ADD COLUMN origin TEXT NOT NULL DEFAULT 'new'")
  data=inputs(str(out))
  with sqlite3.connect('file:'+str(old/'field_calls.sqlite')+'?mode=ro',uri=True) as db:
   for order in (48,96):
    key=f'baseline_o{order}';r=db.execute('SELECT status,input_sha,order_n,started,completed,payload,payload_sha FROM calls WHERE id=?',(key,)).fetchone();h=hashlib.sha256((sha(old/'source_manifest.json')+str(order)).encode())
    for v in (data['centers_mm'],data['frames'],data['sensors_mm']):h.update(np.asarray(v,dtype=np.float64).tobytes())
    if r is None or r[0]!='committed' or r[1]!=h.hexdigest() or r[2]!=order or hashlib.sha256(r[5]).hexdigest()!=r[6]:raise RuntimeError('Archived baseline exact signature/payload')
    with connect(out) as target:target.execute('INSERT INTO calls(id,status,input_sha,order_n,started,completed,payload,payload_sha,origin) VALUES(?,?,?,?,?,?,?,?,?)',(key,*r,'imported'))
 compute_start=time.perf_counter()
 def guard():
  nonlocal peak
  rss=0
  for pr in [psutil.Process()]+psutil.Process().children(recursive=True):
   try:rss+=pr.memory_info().rss
   except psutil.NoSuchProcess:pass
  peak=max(peak,rss/2**20)
  if rss>8*2**30 or psutil.virtual_memory().available<8*2**30 or psutil.disk_usage(str(ROOT)).free<20*2**30 or time.perf_counter()-compute_start>120:raise RuntimeError('Joint magnetic resource/time guard')
 pending=[{'case':case,'nu':p['hypothetical_nu'][0],'load':load} for case in ('inner2.5',) for load in p['load_ids']];results=[]
 pool=cf.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'));active={}
 try:
  while pending or active:
   guard();busy=set(active.values())
   for cpu in (p['assigned_cpu'],):
    if cpu not in busy and pending:active[pool.submit(condition,pending.pop(0),cpu)]=cpu
   save(ROOT/'state.json',{'stage':'cap_coverage_stable_magnetic','observed_at':now(),'conditions_complete':len(results),'scientific_workers':len(active),'RSS_peak_MiB':peak})
   done,_=cf.wait(active,timeout=.2,return_when=cf.FIRST_COMPLETED)
   for f in done:active.pop(f);results.append(f.result())
 except BaseException:
  for pr in psutil.Process().children(recursive=True):
   try:pr.terminate()
   except psutil.NoSuchProcess:pass
  pool.shutdown(wait=False,cancel_futures=True);raise
 else:pool.shutdown(wait=True)
 counts={}
 for label in ('inner2.5',):
  with connect(ROOT/label) as db:counts[label]=[list(r) for r in db.execute('SELECT origin,status,COUNT(*) FROM calls GROUP BY origin,status ORDER BY origin,status')]
  if counts[label]!=[['imported','committed',2],['new','committed',4]]:raise RuntimeError('Six-row same-design field ledger')
 guard();save(ROOT/'aggregate_report.json',{'job_id':p['job_id'],'passed':True,'completed_at_beijing':now(),'conditions':2,'new_field_calls':4,'imported_baselines':2,'case_ledger_counts':counts,'results':results,'maximum_delta_quadrature_gate_ratio':max(r['maximum_delta_quadrature_gate_ratio'] for r in results),'cap_load_change_order_all_resolved':all(r['cap_load_change_order_resolved'] for r in results),'cap_load_change_max_gate_ratio':max(r['cap_load_change_order_gate_ratio_max'] for r in results),'new_FEM_mesh_baseline_computations':0,'compute_wall_seconds':time.perf_counter()-compute_start,'wall_seconds_including_wait':time.perf_counter()-start,'worker_CPU_seconds':sum(r['CPU_seconds'] for r in results),'RSS_peak_MiB':peak,'continuous_coverage_or_spatial_or_material_precision_certified':False})
 save(ROOT/'state.json',{'stage':'finite_stage_complete','observed_at':now(),'conditions_complete':2,'new_field_calls':4})

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save(ROOT/('failure_'+str(time.time_ns())+'.json'),{'at':now(),'error':repr(exc),'traceback':traceback.format_exc()});save(ROOT/'state.json',{'stage':'failed_preserved','observed_at':now(),'error':repr(exc)});raise
