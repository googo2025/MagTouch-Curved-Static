from pathlib import Path
import concurrent.futures as cf,multiprocessing as mp
import json,math,os,sys,time,traceback
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from shell_core import save,exclusive,sha,now

def duffy(n):
 import numpy as np
 x,w=np.polynomial.legendre.leggauss(n);x=(x+1)/2;w=w/2;r,s,t=np.meshgrid(x,x,x,indexing='ij');wr,ws,wt=np.meshgrid(w,w,w,indexing='ij')
 return np.array([r.ravel(),((1-r)*s).ravel(),((1-r)*(1-s)*t).ravel()]),(wr*ws*wt*(1-r)**2*(1-s)).ravel()

def local_operators(nodes,X,W):
 import numpy as np
 from skfem import ElementTetP2,ElementTetP1
 E=ElementTetP2();P=ElementTetP1();d=np.stack([E.lbasis(X,i)[1] for i in range(10)]);pv=np.stack([P.lbasis(X,i)[0] for i in range(4)])
 J=np.einsum('caj,aiq->cqji',nodes-nodes[:,:1],d);det=np.linalg.det(J)
 if np.any(det<=0):raise RuntimeError('Nonpositive integration Jacobian')
 inv=np.linalg.inv(J);g=np.einsum('aiq,cqij->ajcq',d,inv);dx=det*W
 S=np.einsum('aicq,bicq,cq->cab',g,g,dx,optimize=True)
 K=np.einsum('ajcq,bicq,cq->caibj',g,g,dx,optimize=True)+S[:,:,None,:,None]*np.eye(3)[None,None,:,None,:]
 B=-np.einsum('pq,aicq,cq->cpai',pv,g,dx,optimize=True);M=np.einsum('pq,rq,cq->cpr',pv,pv,dx,optimize=True)
 return K.reshape(len(nodes),30,30),B.reshape(len(nodes),4,30),M

def worker(number):
 import numpy as np
 import psutil
 from skfem.quadrature import get_quadrature_tet
 from skfem import MeshTet1,MeshTet2,Basis,ElementVector,ElementTetP2,ElementTetP1,BilinearForm,asm
 from skfem.helpers import ddot,sym_grad,div
 os.sched_setaffinity(0,{4+number});p=json.loads((ROOT/'simulation_plan.json').read_text());st=time.perf_counter();ct=time.process_time();io0=psutil.Process().io_counters();out=ROOT/f'worker{number}';out.mkdir(exist_ok=False)
 with np.load(p['input_files']['selected']['path'],allow_pickle=False) as z:parts=np.array_split(np.arange(len(z['global_tetrahedron_indices'])),4);sel=parts[number];ids=z['global_tetrahedron_indices'][sel];nodes=z['geometry_nodes_mm'][sel]
 exclusive(out/'started.json',{'at':now(),'global_cells':[int(v) for v in ids],'integration_rules':p['integration_rules']})
 matrices={}
 for rule in p['integration_rules']:
  X,W=get_quadrature_tet(int(rule[3:])) if rule.startswith('tet') else duffy(int(rule[5:]));matrices[rule]=local_operators(nodes,X,W)
  np.savez_compressed(out/(rule+'.npz'),global_tetrahedron_indices=ids,K=matrices[rule][0],B=matrices[rule][1],M=matrices[rule][2])
 def rel(a,b):return np.linalg.norm((a-b).reshape(len(a),-1),axis=1)/np.maximum(np.linalg.norm(b.reshape(len(b),-1),axis=1),1e-300)
 ref=matrices['duffy20'];rows=[]
 for j,cell in enumerate(ids):
  rows.append({'global_cell':int(cell),'rule_errors_relative_to_duffy20':{rule:{name:float(rel(a,b)[j]) for name,a,b in zip(('K','B','M'),mats,ref)} for rule,mats in matrices.items()}})
 # Independent library assembly at the same finite rule on one selected cell
 # per batch checks formula/DOF convention, not quadrature convergence.
 nodes0=nodes[:1];m=MeshTet2.from_mesh(MeshTet1(nodes0[0,:4].T,np.arange(4).reshape(4,1)));dofs=m.dofs.element_dofs[:,0];coords=m.doflocs.copy();coords[:,dofs]=nodes0[0].T;m=MeshTet2(coords,m.t);X,W=duffy(8);bu=Basis(m,ElementVector(ElementTetP2()),quadrature=(X,W));bp=Basis(m,ElementTetP1(),quadrature=(X,W))
 @BilinearForm
 def stiff(u,v,w):return 2*ddot(sym_grad(u),sym_grad(v))
 @BilinearForm
 def divergence(u,v,w):return -div(u)*v
 @BilinearForm
 def mass(u,v,w):return u*v
 vector=(3*dofs[:,None]+np.arange(3)[None,:]).ravel();pressure=m.t[:,0];library=(asm(stiff,bu).toarray()[vector][:,vector],asm(divergence,bu,bp).toarray()[pressure][:,vector],asm(mass,bp).toarray()[pressure][:,pressure]);manual=matrices['duffy8'];library_errors={name:float(np.linalg.norm(a-b[0])/max(np.linalg.norm(b[0]),1e-300)) for name,a,b in zip(('K','B','M'),library,manual)}
 local_gates=[]
 for j,(cell,K,M) in enumerate(zip(ids,ref[0],ref[2])):
  rigid=[]
  for mode in range(6):
   tr=np.eye(3)[mode] if mode<3 else np.zeros(3);om=np.zeros(3) if mode<3 else np.eye(3)[mode-3];q=(tr+np.cross(om,nodes[j]-nodes[j].mean(axis=0))).ravel();rigid.append(float(np.linalg.norm(K@q)/max(np.linalg.norm(K)*np.linalg.norm(q),1e-300)))
  eig=np.linalg.eigvalsh(K);meig=np.linalg.eigvalsh(M);local_gates.append({'cell':int(cell),'rigid_null_relative_max':max(rigid),'K_symmetry_relative':float(np.linalg.norm(K-K.T)/np.linalg.norm(K)),'M_symmetry_relative':float(np.linalg.norm(M-M.T)/np.linalg.norm(M)),'K_smallest_eigenvalue_relative':float(eig[0]/eig[-1]),'K_seventh_eigenvalue_relative':float(eig[6]/eig[-1]),'M_smallest_eigenvalue_relative':float(meig[0]/meig[-1])})
 io1=psutil.Process().io_counters();r={'worker':number,'cells':len(ids),'rows':rows,'library_same_rule_relative_errors':library_errors,'local_algebra':local_gates,'worker_wall_seconds':time.perf_counter()-st,'worker_CPU_seconds':time.process_time()-ct,'RSS_end_MiB':psutil.Process().memory_info().rss/2**20,'actual_cpu_affinity':list(os.sched_getaffinity(0)),'numeric_threads':psutil.Process().num_threads(),'IO_bytes':{k:getattr(io1,k)-getattr(io0,k) for k in ('read_bytes','write_bytes')}};save(out/'result.json',r);return r

def main():
 import numpy as np
 import psutil
 p=json.loads((ROOT/'simulation_plan.json').read_text());st=time.perf_counter();ct=time.process_time();peak=0;G=p['gates']
 if str(ROOT)!=p['remote_root'] or Path('/etc/machine-id').read_text().strip()!=p['machine_id']:raise RuntimeError('Machine/root identity')
 for root,pin in [(ROOT,sha(ROOT/'source_manifest.json'))]+[(Path(v['root']),v['source_manifest_sha256']) for v in p['parents']]:
  if sha(root/'source_manifest.json')!=pin:raise RuntimeError('Manifest mismatch')
  for n,h in json.loads((root/'source_manifest.json').read_text())['files'].items():
   if sha(root/n)!=h:raise RuntimeError('Source changed')
 for v in p['parents']:
  root=Path(v['root']);i=json.loads((root/'execution_identity.json').read_text());s=Path(f"/proc/{i['pid']}/stat")
  if any(i[k]!=v[k] for k in ('job_id','pid','proc_start_ticks','source_manifest_sha256')) or s.exists() and s.read_text().rsplit(')',1)[1].split()[19]==i['proc_start_ticks'] or not json.loads((root/'aggregate_report.json').read_text())['passed']:raise RuntimeError('Exact accepted closed parent')
 for v in list(p['input_files'].values())+list(p['installed_source_files'].values()):
  if sha(v['path'])!=v['sha256']:raise RuntimeError('Pinned numerical input/source changed')
 exclusive(ROOT/'execution_identity.json',{'job_id':p['job_id'],'pid':os.getpid(),'proc_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'started_at':now()});os.sched_setaffinity(0,{7})
 def guard():
  nonlocal peak
  rss=0
  for pr in [psutil.Process()]+psutil.Process().children(recursive=True):
   try:rss+=pr.memory_info().rss
   except psutil.NoSuchProcess:pass
  peak=max(peak,rss/2**20)
  if rss>8*2**30 or psutil.virtual_memory().available<8*2**30 or psutil.disk_usage(str(ROOT)).free<20*2**30 or time.perf_counter()-st>p['budget']['wall_seconds_max']:raise RuntimeError('Combined resource/time guard')
 guard();X,W=duffy(4);moment_error=0.
 for a in range(6):
  for b in range(6-a):
   for c in range(6-a-b):
    expected=math.factorial(a)*math.factorial(b)*math.factorial(c)/math.factorial(a+b+c+3);moment_error=max(moment_error,abs(float(np.dot(W,X[0]**a*X[1]**b*X[2]**c))-expected))
 if moment_error>G['reference_monomial_absolute']:raise RuntimeError('Duffy transform qualification')
 save(ROOT/'quadrature_qualification.json',{'Duffy4_all_total_degree_le5_monomials_max_absolute_error':moment_error,'mapping':'x=r,y=(1-r)s,z=(1-r)(1-s)t; J=(1-r)^2(1-s)','new_solve_calls':0})
 pool=cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'));active={pool.submit(worker,i):i for i in range(4)};results=[]
 try:
  while active:
   guard();save(ROOT/'state.json',{'stage':'finite_selected_curved_K_B_M_quadrature','observed_at':now(),'workers_complete':len(results),'RSS_peak_MiB':peak,'new_equilibrium_solves':0})
   done,_=cf.wait(active,timeout=.05,return_when=cf.FIRST_COMPLETED)
   for f in done:active.pop(f);results.append(f.result())
 except BaseException:
  for pr in psutil.Process().children(recursive=True):
   try:pr.terminate()
   except psutil.NoSuchProcess:pass
  pool.shutdown(wait=False,cancel_futures=True);raise
 else:pool.shutdown(wait=True)
 rows=sorted([v for r in results for v in r['rows']],key=lambda v:v['global_cell']);alg=[v for r in results for v in r['local_algebra']];maximum={rule:{name:max(v['rule_errors_relative_to_duffy20'][rule][name] for v in rows) for name in ('K','B','M')} for rule in p['integration_rules']};lib=max(v for r in results for v in r['library_same_rule_relative_errors'].values())
 gates={'reference16_vs20_K':maximum['duffy16']['K']<=G['reference_K_relative'],'reference16_vs20_B_M':max(maximum['duffy16'][n] for n in ('B','M'))<=G['B_M_relative'],'independent_library_assembly':lib<=G['independent_library_relative'],'all_local_algebra':all(v['rigid_null_relative_max']<=G['rigid_null_relative'] and v['K_symmetry_relative']<=G['symmetry_relative'] and v['M_symmetry_relative']<=G['symmetry_relative'] and v['K_smallest_eigenvalue_relative']>=-G['K_negative_eigenvalue_relative'] and v['K_seventh_eigenvalue_relative']>G['K_seventh_eigenvalue_relative_min'] and v['M_smallest_eigenvalue_relative']>G['M_smallest_eigenvalue_relative_min'] for v in alg)}
 choices=[rule for rule in p['candidate_rules'] if maximum[rule]['K']<=G['candidate_K_relative'] and max(maximum[rule][n] for n in ('B','M'))<=G['B_M_relative']];gates['at_least_one_candidate_rule']=bool(choices);passed=bool(all(gates.values()))
 report={'job_id':p['job_id'],'passed':passed,'completed_at_beijing':now(),'gates':gates,'selected_cells':len(rows),'integration_rule_count':len(p['integration_rules']),'local_K_B_M_triples':len(rows)*len(p['integration_rules']),'independent_library_triples':4,'maximum_relative_errors':maximum,'first_all_selected_cells_qualified_candidate_rule':choices[0] if choices and passed else None,'qualified_candidate_rules':choices,'maximum_library_same_rule_error':lib,'reference_moment_max_absolute_error':moment_error,'row_results':rows,'new_equilibrium_RHS_solves_full_matrix_assemblies_meshes_or_fields':0,'wall_seconds':time.perf_counter()-st,'parent_CPU_seconds':time.process_time()-ct,'worker_CPU_seconds':sum(r['worker_CPU_seconds'] for r in results),'RSS_peak_MiB':peak,'worker_resources':[{k:v for k,v in r.items() if k not in ('rows','local_algebra')} for r in results],'full_domain_integration_spatial_material_or_physical_precision_certified':False}
 save(ROOT/'aggregate_report.json',report)
 if not passed:raise RuntimeError('Selected curved operator quadrature gate failed; all rule outputs preserved')
 save(ROOT/'state.json',{'stage':'finite_stage_complete','observed_at':now(),'selected_cells':len(rows),'new_equilibrium_solves':0})
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save(ROOT/('failure_'+str(time.time_ns())+'.json'),{'at':now(),'error':repr(exc),'traceback':traceback.format_exc()});save(ROOT/'state.json',{'stage':'failed_preserved','observed_at':now(),'error':repr(exc)});raise
