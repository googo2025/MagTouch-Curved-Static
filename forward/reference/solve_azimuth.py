"""Finite new diagonal/mid-cap prescribed tractions; no real contact/material claim."""
from pathlib import Path
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from shell_core import save,exclusive,now,sha


def work(index,cpu):
 master=Path(__file__).resolve().parent;top=json.loads((master/"simulation_plan.json").read_text());spec=top["cases"][index];ROOT=master/spec["name"]
 import numpy as np
 from scipy.sparse import load_npz,diags
 from scipy.sparse.linalg import LinearOperator,minres
 from skfem import MeshTet,ElementTetP2
 from skfem.assembly.dofs import Dofs
 p={**top,**spec};os.sched_setaffinity(0,{cpu})
 assigned=p['worker_conditions'][str(index)];selected_ids={s['load'] for s in assigned}
 work_root=ROOT/'workers'/f'worker{index}';work_root.mkdir(parents=True,exist_ok=False)
 exclusive(work_root/'started.json',{'at':now(),'index':index,'cpu':cpu,'assigned':assigned})
 import psutil
 io_before=psutil.Process().io_counters()
 sys.path.insert(0,p['dependency_root']);import pyamg
 if pyamg.__version__!='5.3.0' or Path(p['dependency_root']) not in Path(pyamg.__file__).parents:raise RuntimeError('Qualified isolated AMG package mismatch')
 start,ct=time.perf_counter(),time.process_time();pins=json.loads((ROOT/'input_identity.json').read_text())
 for n,path in p['inputs'].items():
  if sha(Path(path))!=pins[n]:raise RuntimeError('Frozen input changed')
 with np.load(p['inputs']['proposal'],allow_pickle=False) as a:loc=a['proposed_geometry_nodes_mm'];tets=a['oriented_original_tetrahedra'];domain=a['magnet_source_index']
 with np.load(p['inputs']['identity'],allow_pickle=False) as a:free=a['matrix_free_scalar_P2_dofs'];centers=a['body_centers_mm'];owner=a['scalar_P2_body_owner']
 from skfem import MeshTet2,FacetBasis,ElementTetP1
 mesh=MeshTet2(loc.T,tets.T)
 with np.load(p['inputs']['numbering'],allow_pickle=False) as a:ed=a['scalar_P2_element_dofs']
 if not np.array_equal(mesh.dofs.element_dofs,ed):raise RuntimeError('Qualified exact curved scalar mapping')
 boundary=mesh.boundary_facets();verts=loc[mesh.facets[:,boundary]];base_annulus=boundary[np.all(np.abs(verts[:,:,2])<1e-10,axis=0)];clamp_scalar=np.unique(Dofs(mesh,ElementTetP2()).get_facet_dofs(base_annulus).all())
 if len(base_annulus)!=100 or len(clamp_scalar)!=248 or np.any(owner[clamp_scalar]>=0):raise RuntimeError('Qualified new base support identity')
 if sha(Path(p['inputs']['accepted_base_area']))!=p['saved_output_SHA256']['base_area']:raise RuntimeError('Accepted base-area source digest')
 ar=json.loads(Path(p['inputs']['accepted_base_area']).read_text());areas=ar['area_mm2'];area=areas[-1];analytic=math.pi*(20**2-14**2)
 if not ar['passed'] or ar['facets']!=100 or ar['orders']!=[8,10] or abs(areas[0]-area)/area>1e-8 or abs(area-analytic)/analytic>.05:raise RuntimeError('Original accepted curved base area gates')
 inverse=np.full(len(loc),-1);inverse[free]=np.arange(len(free));clamp=(3*inverse[clamp_scalar,None]+np.arange(3)).ravel()
 K=load_npz(p['inputs']['K']);B=load_npz(p['inputs']['B']);M=load_npz(p['inputs']['M']);nfree=3*len(free)
 unknown=np.setdiff1d(np.arange(K.shape[0]),clamp);K=K[unknown][:,unknown].tocsr();B=B[:,unknown].tocsr();nu_dof=K.shape[0];np_dof=M.shape[0]
 if nu_dof%3 or nu_dof+np_dof>700000:raise RuntimeError('Three-component BSR/mixed ceiling')
 save(work_root/'clamp_identity.json',{'base_annulus_facets':len(base_annulus),'base_annulus_area_mm2':area,'analytic_base_annulus_area_mm2':analytic,'area8_10_relative':abs(areas[0]-area)/area,'clamped_scalar_P2_DOF':len(clamp_scalar),'free_displacement_DOF':nu_dof,'pressure_DOF':np_dof,'geometry':p['name']+' accepted repaired curved geometry','base_area_integrals_reused':True})
 candidates=[]
 for mode in range(6):
  t=np.eye(3)[mode] if mode<3 else np.zeros(3);omega=np.zeros(3) if mode<3 else 1e-3*np.eye(3)[mode-3]
  v=np.empty(nfree+930);v[:nfree]=(t+np.cross(omega,loc[free])).ravel()
  for body in range(155):v[nfree+6*body:nfree+6*body+6]=np.r_[t+np.cross(omega,centers[body]),omega]
  candidates.append(v[unknown])
 scale=1/np.sqrt(K.diagonal());Ks=(diags(scale)@K@diags(scale)).tobsr(blocksize=(3,3));C=np.column_stack(candidates)/scale[:,None]
 np.random.seed(20261106)
 save(work_root/'state.json',{'stage':'building_one_scaled_BSR_AMG','observed_at':now(),'algebraic_solves_submitted':0})
 ml=pyamg.smoothed_aggregation_solver(Ks,B=C,symmetry='symmetric',max_levels=10,max_coarse=30,
    presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}))
 complexity=float(ml.operator_complexity())
 if complexity>p['gates']['operator_complexity_max']:raise RuntimeError('AMG operator complexity guard')
 Pinv=ml.aspreconditioner(cycle='V');mass_lump=np.asarray(M.sum(axis=1)).ravel()
 if np.any(mass_lump<=0):raise RuntimeError('Positive pressure mass lump required')
 from scipy.spatial.transform import Rotation
 with np.load(p['inputs']['mesh'],allow_pickle=False) as saved:nominal=saved['nominal_centers_mm'];frames=saved['nominal_frames']
 if not np.array_equal(nominal,centers):raise RuntimeError('Protected source nominal centers mismatch')
 from skfem.quadrature import get_quadrature_tet
 from quadrature_core import duffy
 with np.load(p['inputs']['identity'],allow_pickle=False) as a:oldloc=a['scalar_P2_dof_locations_mm']
 affected=np.any(np.any(loc!=oldloc,axis=1)[ed],axis=0);matrixids=np.flatnonzero(domain==-1)
 def strain_check(unodes):
  maximum=0.;E=ElementTetP2()
  for ids,X in ((matrixids,get_quadrature_tet(4)[0]),(np.flatnonzero((domain==-1)&affected),duffy(8)[0])):
   d=np.stack([E.lbasis(X,i)[1] for i in range(10)])
   for offset in range(0,len(ids),256):
    chosen=ids[offset:offset+256];en=loc[ed[:,chosen].T];u=unodes[ed[:,chosen].T];J=np.einsum('caj,aiq->cqji',en-en[:,:1],d);inv=np.linalg.inv(J);g=np.einsum('aiq,cqij->ajcq',d,inv);gradient=np.einsum('caj,aicq->jicq',u,g);strain=.5*(gradient+gradient.swapaxes(0,1))
    if not np.isfinite(strain).all():raise RuntimeError('Nonfinite actual curved physical strain')
    maximum=max(maximum,float(np.sqrt(np.sum(strain*strain,axis=(0,1))).max()))
  return maximum
 del oldloc
 operator_report=json.loads((Path(p['operator_parent']['root'])/'aggregate_report.json').read_text())
 for name in ('K','B','M'):
  if pins[name]!=operator_report['operators'][name]['sha256']:raise RuntimeError('Accepted curved operator digest')
 Kall=load_npz(p['inputs']['K']);Ball=load_npz(p['inputs']['B']);T=load_npz(p['inputs']['constraint'])
 from skfem import ElementVector,LinearForm,asm
 from skfem.helpers import dot
 vertices=loc[mesh.facets[:,boundary]]
 radius=np.where(vertices[:,:,2]<=100+1e-9,np.linalg.norm(vertices[:,:,:2],axis=2),np.linalg.norm(vertices-np.array([0,0,100]),axis=2))
 outer=boundary[np.all(np.abs(radius-20)<1e-6,axis=0)]
 if not len(outer):raise RuntimeError('Outer surface identity')
 outer_area=float(FacetBasis(mesh,ElementTetP1(),facets=outer,intorder=10,disable_doflocs=True).dx.sum());analytic_outer=2*math.pi*20*100+2*math.pi*20**2
 if abs(outer_area-analytic_outer)/analytic_outer>.05:raise RuntimeError('Original outer area geometry gate')
 loads={};load_reports=[]
 for setting in p['loads']:
  if setting['id'] not in selected_ids:continue
  center=np.asarray(setting['center_mm']);axis=np.asarray(setting['inward_axis'])
  radial=center.copy();radial[2]=max(center[2]-100.,0.)
  if abs(np.linalg.norm(radial)-20)>1e-10 or abs(np.linalg.norm(axis)-1)>1e-12 or np.linalg.norm(radial/20+axis)>1e-12:raise RuntimeError('Frozen outer center/inward-axis geometry contract')
  sigma_mm=float(setting['Gaussian_sigma_mm'])
  if sigma_mm != 12.0:raise RuntimeError('Frozen load width')
  order_forces={}
  for order in (8,10):
   fb=FacetBasis(mesh,ElementVector(ElementTetP2()),facets=outer,intorder=order,disable_doflocs=True)
   @LinearForm
   def traction(v,w):return np.exp(-np.sum((w.x-center[:,None,None])**2,axis=0)/(2*sigma_mm**2))*dot(-w.n,v)
   value=asm(traction,fb);projection=float(value.reshape(-1,3).sum(axis=0)@axis)
   if projection<=0:raise RuntimeError('Positive Gaussian load normalization required')
   value*=.01/projection;order_forces[order]=value;del fb
  quadrature_relative=float(np.linalg.norm(order_forces[8]-order_forces[10])/np.linalg.norm(order_forces[10]))
  if quadrature_relative>1e-8:raise RuntimeError('Refined-grid8-to10 surface integration gate failed')
  f=order_forces[10];fr=np.asarray(T.T@f)
  total=f.reshape(-1,3).sum(axis=0);torque=np.cross(loc,f.reshape(-1,3)).sum(axis=0)
  identity={'load':setting['id'],'Gaussian_sigma_mm':sigma_mm,'integration_order':10,'order8_to10_relative':quadrature_relative,'axis_force_N':float(total@axis),'total_force_N':total.tolist(),
    'total_torque_Nmm':torque.tolist(),'load_vector_sha256':__import__('hashlib').sha256(f.tobytes()).hexdigest()}
  if abs(identity['axis_force_N']-.01)>1e-12:raise RuntimeError('Load normalization gate')
  np.savez_compressed(work_root/f"load_{setting['id']}_vectors.npz",full_force_N=f,reduced_force_N=fr)

  loads[setting['id']]=(f,fr,identity);load_reports.append(identity)
 save(work_root/'load_identity.json',{'outer_area_mm2':outer_area,'analytic_outer_area_mm2':analytic_outer,
      'outer_facets':len(outer),'load_assemblies':len(load_reports),'loads':load_reports})
 results=[];rng=np.random.default_rng(20261106)
 for nu in p['hypothetical_nu']:
  mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu))
  def action(v):return np.r_[mu*(K@v[:nu_dof])+B.T@v[nu_dof:],B@v[:nu_dof]-(M@v[nu_dof:])/lam]
  def precondition(v):return np.r_[scale*(Pinv@(scale*v[:nu_dof]))/mu,v[nu_dof:]/((1/mu+1/lam)*mass_lump)]
  A=LinearOperator((nu_dof+np_dof,nu_dof+np_dof),matvec=action,rmatvec=action,dtype=float)
  P=LinearOperator(A.shape,matvec=precondition,rmatvec=precondition,dtype=float)
  x,y=rng.normal(size=(2,A.shape[0]));px,py=P@x,P@y
  symmetry=abs(x@py-y@px)/max(abs(x@py),abs(y@px),1e-30)
  if symmetry>1e-10 or x@px<=0 or y@py<=0:raise RuntimeError('Physical mixed preconditioner gate')
  for setting in p['loads']:
   if {'nu':nu,'load':setting['id']} not in assigned:continue
   load=setting['id'];old_iterations=0;f,fr,load_info=loads[load];rhs=np.r_[fr[unknown],np.zeros(np_dof)]
   out=ROOT/f'nu{nu:g}_{load}';out.mkdir(exist_ok=True)
   archived=p['recovery_archives'].get(load)
   if archived is not None:
    import shutil
    folder=Path(archived['case_root'])
    files=archived['sha256']
    if set(files)!={'result.json','physical_solution.npz','rigid_source_motion.npz'}:
     raise RuntimeError('Incomplete recovery pins')
    if any(sha(folder/name)!=pin for name,pin in files.items()):
     raise RuntimeError('Archived mechanics payload SHA')
    report=json.loads((folder/'result.json').read_text())
    if report['load']!=load or report['nu']!=nu or not report['passed'] or not all(report['gates'].values()) or report['load_identity']!=load_info:
     raise RuntimeError('Archived mechanics exact load and original gates')
    for name in ('physical_solution.npz','rigid_source_motion.npz'):
     with (folder/name).open('rb') as src,(out/name).open('xb') as dst:
      shutil.copyfileobj(src,dst)
     if sha(out/name)!=files[name]:raise RuntimeError('Archived mechanics copy SHA')
    report=dict(report,new_physical_FEM_RHS=0,continued_physical_FEM_RHS=0,reused_physical_FEM_RHS=1,recovery_archive=archived)
    save(out/'result.json',report);results.append(report)
    continue
   exclusive(out/'solve_started.json',{'at':now(),'nu':nu,'load':load,'maxiter_per_correction':1000,'new_physical_FEM_RHS':1,'historical_iterations':0,'old_flat_initial_guess_prior_iterations':old_iterations,'equations_max':3,'cold_initial_state':True})
   case_out=out;iterations=[0];history=[];case_start=time.perf_counter();normrhs=np.linalg.norm(rhs)
   solution=np.zeros(nu_dof+np_dof);before=float(np.linalg.norm(rhs-A@solution)/normrhs)
   exclusive(case_out/'cold_initial_state.json',{'at':now(),'new_physical_FEM_RHS':1,'historical_iterations':0,'cold_initial_state':True,'equations_max':3})
   def evaluate(solution,info):
    true_residual=float(np.linalg.norm(rhs-A@solution)/normrhs)
    qfull=np.zeros(Kall.shape[0]);qfull[unknown]=solution[:nu_dof];pressure=solution[nu_dof:]
    full_u=np.asarray(T@qfull);rd=mu*(Kall@qfull)+Ball.T@pressure-fr;rp=Ball@qfull-(M@pressure)/lam
    reaction=rd[clamp].reshape(-1,3);external_force=f.reshape(-1,3).sum(axis=0)
    external_torque=np.cross(loc,f.reshape(-1,3)).sum(axis=0)
    force_balance=float(np.linalg.norm(reaction.sum(axis=0)+external_force)/.01)
    torque_balance=float(np.linalg.norm(np.cross(loc[clamp_scalar],reaction).sum(axis=0)+external_torque)/(.01*20))
    body_res=rd[nfree:].reshape(155,6)
    body_force=float(np.linalg.norm(body_res[:,:3],axis=1).max()/.01)
    body_torque=float(np.linalg.norm(body_res[:,3:],axis=1).max()/(.01*2.5))
    work_load=float(qfull@fr);stored_twice=mu*float(qfull@(Kall@qfull))+float(pressure@(M@pressure))/lam
    energy=abs(stored_twice-work_load)/max(abs(work_load),1e-30)
    full_u_nodes=full_u.reshape(-1,3)
    maxstrain=strain_check(full_u_nodes)
    body_pose=qfull[nfree:].reshape(155,6);translation=body_pose[:,:3];omega=body_pose[:,3:]
    maxrotation=float(np.linalg.norm(omega,axis=1).max())
    moved_centers=centers+translation;moved_frames=Rotation.from_rotvec(omega).as_matrix()@frames
    np.savez_compressed(out/'physical_solution.npz',reduced_displacement=qfull,pressure_matrix=pressure,
        full_displacement_mm=full_u,history=history)
    np.savez_compressed(out/'rigid_source_motion.npz',source_centers_mm=centers,nominal_frames=frames,
        displacement_mm=translation,rotation_vectors_rad=omega,moved_centers_mm=moved_centers,moved_frames=moved_frames,
        linear_infinitesimal_rigid_coordinates=body_pose)
    gates={'solver_info':info==0,'true_free_residual':true_residual<=1e-8,'global_force':force_balance<=1e-8,
     'global_torque':torque_balance<=1e-8,'all_body_force':body_force<=1e-8,'all_body_torque':body_torque<=1e-8,
     'energy':energy<=1e-8,'positive_work':work_load>0,'small_strain':maxstrain<=.01,'small_rigid_rotation':maxrotation<=.01,
     'finite':bool(np.isfinite(solution).all() and np.isfinite(maxstrain))}
    report={'nu':nu,'load':load,'passed':all(gates.values()),'gates':gates,'MINRES_info':int(info),'iterations':iterations[0],
     'true_relative_residual':true_residual,'global_force_balance_ratio':force_balance,'global_torque_balance_ratio':torque_balance,
     'maximum_body_force_ratio':body_force,'maximum_body_torque_ratio':body_torque,'energy_work_ratio':energy,
     'load_work_Nmm':work_load,'compliance_mm_per_N':work_load/.01**2,'maximum_matrix_strain_Frobenius':maxstrain,
     'maximum_rigid_rotation_rad':maxrotation,'maximum_source_translation_mm':float(np.linalg.norm(translation,axis=1).max()),
     'maximum_P2_displacement_mm':float(np.linalg.norm(full_u_nodes,axis=1).max()),'preconditioner_symmetry_relative':float(symmetry),
     'load_identity':load_info,'wall_seconds':time.perf_counter()-case_start,'new_physical_FEM_RHS':1,'continued_physical_FEM_RHS':0,'historical_iterations':0,'old_flat_initial_guess_prior_iterations':old_iterations,'cumulative_iterations':iterations[0],'correction_solves_completed':correction_step,
     'measured_material_or_contact':False,'nonlinear_silicone_or_geometry':False}
    return report
   for correction_step in (1,2,3):
    out=case_out/f'correction{correction_step:02d}';out.mkdir(exist_ok=False)
    residual_rhs=rhs-A@solution
    exclusive(out/'started.json',{'at':now(),'step':correction_step,'historical_iterations':0,'old_flat_initial_guess_prior_iterations':old_iterations,'new_iterations_before_step':iterations[0],'zero_start':True,'maxiter':1000,'rtol':1e-11,'initial_original_true_relative_residual':float(np.linalg.norm(residual_rhs)/normrhs)})
    local_iterations=[0]
    def callback(v):
     iterations[0]+=1;local_iterations[0]+=1
     if local_iterations[0]%10==0:
      relative=float(np.linalg.norm(rhs-A@(solution+v))/normrhs);history.append([iterations[0],relative])
      correction_relative=float(np.linalg.norm(residual_rhs-A@v)/np.linalg.norm(residual_rhs))
      progress={'at':now(),'iterations':iterations[0],'correction_step':correction_step,'correction_iterations':local_iterations[0],'historical_iterations':0,'old_flat_initial_guess_prior_iterations':old_iterations,'cumulative_iterations':iterations[0],'true_relative_residual':relative,'correction_rhs_relative_residual':correction_relative}
      save(case_out/'progress.json',progress)
      save(work_root/'state.json',{'stage':'curved_new_problem_residual_equation','observed_at':now(),'worker':index,'nu':nu,'load':load,**progress})
    delta,info=minres(A,residual_rhs,M=P,rtol=1e-11,maxiter=1000,callback=callback,check=True)
    np.savez_compressed(out/'correction_checkpoint.npz',initial_solution=solution,correction=delta,original_rhs_residual=residual_rhs,corrected_solution=solution+delta)
    solution=solution+delta;report=evaluate(solution,info)
    report['correction_iterations']=local_iterations[0];report['initial_true_relative_residual_before_phase']=before
    save(out/'result.json',report)
    if report['passed'] or info!=0:break
   import shutil
   for filename in ('physical_solution.npz','rigid_source_motion.npz'):
    with (out/filename).open('rb') as src,(case_out/filename).open('xb') as dst:shutil.copyfileobj(src,dst)
   save(case_out/'result.json',report);results.append(report)
   if not report['passed']:raise RuntimeError('Unchanged original physical gates failed after bounded residual correction; preserve all steps')
 return {'job_id':p['job_id'],'passed':True,'completed_at_beijing':now(),'results':results,'conditions':len(results),
  'AMG_hierarchies':1,'AMG_levels':len(ml.levels),'AMG_operator_complexity':complexity,'new_physical_FEM_RHS':sum(r['new_physical_FEM_RHS'] for r in results),'continued_physical_FEM_RHS':0,'correction_solves':sum(r['correction_solves_completed'] for r in results if r['new_physical_FEM_RHS']),
  'new_volume_FEM_assemblies':0,'surface_load_assemblies':2*len(load_reports),'new_mesh_or_field_or_training_calls':0,
  'whole_155_rigid_inclusion_linear_traction_solved':True,'material_calibrated':False,'actual_contact_solved':False,
  'spatial_convergence_certified':False,'physical_sensor_precision_certified':False,
  'wall_seconds_worker':time.perf_counter()-start,'CPU_seconds_worker':time.process_time()-ct,'worker':index,'cpu':cpu,
  'io_read_bytes_delta':psutil.Process().io_counters().read_bytes-io_before.read_bytes,
  'io_write_bytes_delta':psutil.Process().io_counters().write_bytes-io_before.write_bytes}


def main():
 import psutil
 p=json.loads((ROOT/'simulation_plan.json').read_text());start,ct=time.perf_counter(),time.process_time();peak=0
 if Path('/etc/machine-id').read_text().strip()!=p['machine_id'] or str(ROOT)!=p['remote_root']:raise RuntimeError('Machine/root')
 for parent in [{'root':str(ROOT),'source_manifest_sha256':sha(ROOT/'source_manifest.json')}]+p['parents']:
  r=Path(parent['root'])
  if sha(r/'source_manifest.json')!=parent['source_manifest_sha256']:raise RuntimeError('Manifest')
  for n,h in json.loads((r/'source_manifest.json').read_text())['files'].items():
   if Path(n).name!=n or sha(r/n)!=h:raise RuntimeError('Immutable source')
  if r!=ROOT:
   i=json.loads((r/'execution_identity.json').read_text())
   if any(i[k]!=parent[k] for k in ('job_id','pid','proc_start_ticks','source_manifest_sha256')):raise RuntimeError('Exact accepted input identity')
   st=Path(f"/proc/{i['pid']}/stat")
   if st.exists() and st.read_text().rsplit(')',1)[1].split()[19]==i['proc_start_ticks']:raise RuntimeError('Input parent active')
   if not json.loads((r/'aggregate_report.json').read_text())['passed']:raise RuntimeError('Input parent not accepted')
 exclusive(ROOT/'execution_identity.json',{'job_id':p['job_id'],'pid':os.getpid(),'proc_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'started_at':now()})
 for item in p['installed_source_files'].values():
  if sha(item['path'])!=item['sha256']:raise RuntimeError('Installed numerical source')
 os.sched_setaffinity(0,{7})
 cores=[(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text().strip(),Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text().strip()) for c in (4,5,6,7)]
 if len(set(cores))!=4:raise RuntimeError('Four physical cores')
 for spec in p['cases']:
  r=ROOT/spec['name'];r.mkdir(exist_ok=False)
  for n,path in spec['inputs'].items():
   if sha(Path(path))!=spec['input_pins'][n]:raise RuntimeError('Frozen input '+n)
  save(r/'input_identity.json',spec['input_pins'])
 def guard():
  nonlocal peak
  total=0
  for pr in [psutil.Process()]+psutil.Process().children(recursive=True):
   try:
    rss=pr.memory_info().rss;total+=rss
    if pr.pid!=os.getpid() and rss>3.5*2**30:raise RuntimeError('Worker RSS')
   except psutil.NoSuchProcess:pass
  peak=max(peak,total/2**20)
  if total>8*2**30 or psutil.virtual_memory().available<8*2**30 or psutil.disk_usage(str(ROOT)).free<20*2**30 or time.perf_counter()-start>7200:raise RuntimeError('Joint resource/time guard')
 guard()
 if psutil.virtual_memory().available<15.1*2**30:raise RuntimeError('Two3.5GiB reservations and8GiB system headroom')
 save(ROOT/'resource_preflight.json',{'at':now(),'physical_core_pairs':cores,'workers':2,'cpus':[4,6],'numeric_threads':1,'aggregate_RSS_GiB':8,'per_worker_reservation_GiB':3.5,'GPU':False})
 pool=cf.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'));active={pool.submit(work,i,s['cpu']):s['name'] for i,s in enumerate(p['cases'])};reports=[]
 try:
  while active:
   guard();progress=[]
   for spec in p['cases']:
    r=ROOT/spec['name'];q=r/'workers/worker0/state.json';q2=r/'workers/worker1/state.json'
    for q in (q,q2):
     if q.exists():progress.append({'case':spec['name'],**json.loads(q.read_text())})
   complete=list(ROOT.glob('inner*/nu*/result.json'))
   save(ROOT/'state.json',{'stage':'curved_azimuth_cold_physical_and_residual_equations','observed_at':now(),'conditions_total':2,'conditions_with_results':len(complete),'active_cases':list(active.values()),'worker_states':progress,'RSS_peak_MiB':peak})
   done,_=cf.wait(active,timeout=.5,return_when=cf.FIRST_COMPLETED)
   for f in done:
    label=active.pop(f);result=f.result();result['case']=label;save(ROOT/label/'aggregate_report.json',result);reports.append(result)
 except BaseException:
  for pr in psutil.Process().children(recursive=True):
   try:pr.terminate()
   except psutil.NoSuchProcess:pass
  pool.shutdown(wait=False,cancel_futures=True);raise
 else:pool.shutdown(wait=True)
 rows=[dict(r,case=v['case']) for v in reports for r in v['results']]
 if len(rows)!=2 or not all(r['passed'] for r in rows) or sum(r['iterations'] for r in rows)>6000:raise RuntimeError('Two bounded conditions')
 guard();save(ROOT/'aggregate_report.json',{'job_id':p['job_id'],'passed':True,'completed_at_beijing':now(),'conditions':2,'new_physical_FEM_RHS':sum(r['new_physical_FEM_RHS'] for r in rows),'reused_physical_FEM_RHS':sum(r.get('reused_physical_FEM_RHS',0) for r in rows),'new_surface_load_vectors':4,'new_FEM_volume_assemblies_mesh_field_training':0,'results':rows,'case_results':reports,'AMG_hierarchies':1,'wall_seconds':time.perf_counter()-start,'parent_CPU_seconds':time.process_time()-ct,'worker_CPU_seconds':sum(r['CPU_seconds_worker'] for r in reports),'RSS_peak_MiB':peak,'candidate_adopted':False,'spatial_or_real_material_precision_certified':False})
 save(ROOT/'state.json',{'stage':'finite_stage_complete','observed_at':now(),'conditions_complete':2,'new_physical_FEM_RHS':sum(r['new_physical_FEM_RHS'] for r in rows)})

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save(ROOT/('failure_'+str(time.time_ns())+'.json'),{'at':now(),'error':repr(exc),'traceback':traceback.format_exc()});save(ROOT/'state.json',{'stage':'failed_preserved','observed_at':now(),'error':repr(exc)});raise
