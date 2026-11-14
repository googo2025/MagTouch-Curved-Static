"""Remote-only homogeneous linear shell traction baseline; not a contact solver."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import time

def native(value):
    # NumPy scalar metadata are converted without changing scientific arrays.
    if hasattr(value,'item'):return value.item()
    raise TypeError('Unsupported result metadata type: '+type(value).__name__)

def save(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False,default=native)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def exclusive(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False,default=native)
        f.flush();os.fsync(f.fileno())

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mesh_block(root_text,h,cpu):
    import numpy as np
    from scipy.sparse import bmat
    from scipy.sparse.linalg import splu
    from skfem import MeshTet,Basis,FacetBasis,ElementVector,ElementTetP2,ElementTetP1,BilinearForm,LinearForm,asm
    from skfem.helpers import ddot,sym_grad,div,dot
    root=Path(root_text);plan=json.loads((root/'simulation_plan.json').read_text())
    os.sched_setaffinity(0,{cpu})
    block=root/'blocks'/f'h{h:g}';block.mkdir(parents=True,exist_ok=True)
    exclusive(block/'started.json',{'h':h,'cpu':cpu,'pid':os.getpid(),'time':now()})
    start,ct=time.perf_counter(),time.process_time()
    source=Path(plan['mesh_parent']['root'])/'tasks'/f'capsule_mesh_h{h:g}'/'mesh.npz'
    mesh_sha=sha(source)
    with np.load(source,allow_pickle=False) as saved:
        nodes=saved['nodes_mm'];tets=saved['tetrahedra']
        mesh=MeshTet(nodes.T,tets.T,sort_t=False)
        if not np.array_equal(mesh.p,nodes.T) or not np.array_equal(mesh.t,tets.T):raise RuntimeError('Stored mesh indexing changed')
    if sha(source)!=mesh_sha:raise RuntimeError('Stored mesh changed during read')
    boundary=mesh.boundary_facets();vertices=mesh.p[:,mesh.facets[:,boundary]]
    length=plan['geometry']['cylinder_length_mm'];ri=plan['geometry']['inner_radius_mm'];ro=plan['geometry']['outer_radius_mm']
    radius=np.sqrt(vertices[0]**2+vertices[1]**2+np.maximum(vertices[2]-length,0.)**2)
    tol=plan['acceptance']['boundary_node_absolute_mm_max']
    bottom=np.all(np.abs(vertices[2])<=tol,axis=0)
    inner=np.all(np.abs(radius-ri)<=tol,axis=0)&~bottom
    outer=np.all(np.abs(radius-ro)<=tol,axis=0)&~bottom
    if not np.all(inner.astype(int)+outer.astype(int)+bottom.astype(int)==1) or not all(np.any(x) for x in (inner,outer,bottom)):
        raise RuntimeError('Boundary classification incomplete or overlapping')
    inner_facets=boundary[inner];outer_facets=boundary[outer];bottom_facets=boundary[bottom]
    order=plan['model']['integration_order'];element=ElementVector(ElementTetP2())
    bu=Basis(mesh,element,intorder=order);bp=Basis(mesh,ElementTetP1(),intorder=order)
    fb=FacetBasis(mesh,element,facets=outer_facets,intorder=order)
    D=bu.get_dofs(facets=inner_facets).all();components=bu.split_indices()
    total=int(bu.N+bp.N)
    if total>plan['budget']['mixed_DOF_max_per_mesh']:raise RuntimeError('Finite DOF budget exceeded')
    free=np.setdiff1d(np.arange(total),D)
    coords=bu.doflocs[:,components[0]]
    if any(not np.array_equal(coords,bu.doflocs[:,c]) for c in components):raise RuntimeError('Component coordinate mismatch')
    @BilinearForm
    def unit_elasticity(u,v,w):return 2*ddot(sym_grad(u),sym_grad(v))
    @BilinearForm
    def coupling(u,q,w):return -q*div(u)
    @BilinearForm
    def mass(p,q,w):return p*q
    A0=asm(unit_elasticity,bu);B=asm(coupling,bu,bp);M=asm(mass,bp)
    E=plan['model']['E_normalization_N_per_mm2'];target=plan['load']['projected_resultant_N']
    loads={}
    for setting in plan['load']['positions']:
        center=np.array(setting['center_mm']);direction=np.array(setting['inward_axis'])
        width=plan['load']['Gaussian_sigma_mm']
        @LinearForm
        def pressure(v,w):
            distance2=np.sum((w.x-center[:,None,None])**2,axis=0)
            return -np.exp(-distance2/(2*width**2))*dot(w.n,v)
        raw=asm(pressure,fb)
        raw_resultant=np.array([float(raw[c].sum()) for c in components])
        projection=float(direction@raw_resultant)
        if not projection>0.:raise RuntimeError('Inward load projection not positive')
        factor=target/projection;F=raw*factor
        resultant=np.array([float(F[c].sum()) for c in components])
        loads[setting['id']]={'F':F,'factor':factor,'resultant':resultant,'direction':direction,'setting':setting}
    save(block/'geometry_and_loads.json',{'h':h,'mesh_sha256':mesh_sha,'nodes':mesh.nvertices,'tetrahedra':mesh.nelements,
        'displacement_DOF':int(bu.N),'pressure_DOF':int(bp.N),'clamped_DOF':len(D),
        'boundary_facets':{'inner':len(inner_facets),'outer':len(outer_facets),'bottom_free':len(bottom_facets)},
        'loads':{k:{'pressure_scale_N_per_mm2':v['factor'],'resultant_N':v['resultant'].tolist()} for k,v in loads.items()}})
    results=[]
    for nu in plan['model']['hypothetical_nu']:
        mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
        A=mu*A0;K=bmat([[A,B.T],[B,-M/lam]],format='csr')
        delta=K-K.T
        asym=float(np.max(np.abs(delta.data),initial=0.)/np.max(np.abs(K.data)))
        if asym>plan['acceptance']['relative_matrix_asymmetry_max']:raise RuntimeError('Mixed matrix asymmetric')
        factor_dir=block/f'nu{nu:g}';factor_dir.mkdir(exist_ok=True)
        exclusive(factor_dir/'factorization_started.json',{'time':now(),'mixed_DOF':total,'sparse_nonzeros':K.nnz})
        factor_start=time.perf_counter();lu=splu(K[free][:,free].tocsc())
        save(factor_dir/'factorization_complete.json',{'seconds':time.perf_counter()-factor_start,'L_nonzeros':lu.L.nnz,'U_nonzeros':lu.U.nnz})
        for load_id,load in loads.items():
            out=factor_dir/load_id;out.mkdir(exist_ok=True)
            exclusive(out/'started.json',{'time':now(),'h':h,'nu':nu,'load':load['setting'],'FEM_rhs_solve':1})
            ts,tc=time.perf_counter(),time.process_time()
            rhs=np.concatenate([load['F'],np.zeros(bp.N)])
            solution=np.zeros(total);solution[free]=lu.solve(rhs[free])
            u=solution[:bu.N];p=solution[bu.N:];residual=K@solution-rhs
            scale=max(float(np.max(np.abs(K).sum(axis=1)))*float(np.max(np.abs(solution))),target,1e-300)
            residual_scaled=float(np.max(np.abs(residual[free]),initial=0.)/scale)
            reaction=np.zeros(bu.N);reaction[D]=residual[D]
            reaction_vector=np.vstack([reaction[c] for c in components]);load_vector=np.vstack([load['F'][c] for c in components])
            reaction_force=reaction_vector.sum(axis=1);applied=load_vector.sum(axis=1)
            net_force=float(np.linalg.norm(reaction_force+applied)/target)
            moment_reaction=np.cross(coords.T,reaction_vector.T).sum(axis=0)
            moment_load=np.cross(coords.T,load_vector.T).sum(axis=0)
            moment_error=float(np.linalg.norm(moment_reaction+moment_load)/(target*(length+ro)))
            work=float(load['F']@u);energy=float(.5*u@(A@u+B.T@p))
            energy_error=abs(2*energy-work)/max(abs(work),1e-300)
            uq=bu.interpolate(u);grad=np.asarray(uq.grad);strain=.5*(grad+np.swapaxes(grad,0,1))
            strain_frobenius=np.sqrt(np.sum(strain**2,axis=(0,1)))
            max_strain=float(np.max(strain_frobenius));divu=np.trace(grad,axis1=0,axis2=1)
            pq=np.asarray(bp.interpolate(p));weak_pressure_rms=float(np.sqrt(np.sum((pq+lam*divu)**2*bp.dx)/np.sum(bp.dx)))
            unodal=np.vstack([u[c] for c in components]);umax=float(np.max(np.linalg.norm(unodal,axis=0)))
            compliance=work/target**2
            projection_error=abs(float(load['direction']@applied)-target)/target
            t=plan['acceptance']
            gates={'finite_solution':bool(np.isfinite(solution).all()),'positive_work':work>0.,
                'free_residual':residual_scaled<=t['free_residual_scaled_max'],
                'force_balance':net_force<=t['force_balance_relative_max'],
                'moment_balance':moment_error<=t['moment_balance_relative_max'],
                'energy_work':energy_error<=t['energy_work_relative_max'],
                'projected_load':projection_error<=t['projected_load_relative_max'],
                'small_strain':max_strain<=t['max_quadrature_strain_Frobenius']}
            metrics={'compliance_mm_per_N_at_normalized_E':compliance,'weighted_patch_displacement_mm':work/target,
                'max_nodal_displacement_mm':umax,'max_quadrature_strain_Frobenius':max_strain,
                'free_residual_scaled':residual_scaled,'force_balance_relative':net_force,'moment_balance_relative':moment_error,
                'energy_work_relative':energy_error,'projected_load_relative':projection_error,
                'work_N_mm':work,'strain_energy_N_mm':energy,'applied_resultant_N':applied.tolist(),
                'support_reaction_N':reaction_force.tolist(),'applied_moment_N_mm':moment_load.tolist(),
                'support_moment_N_mm':moment_reaction.tolist(),'pressure_constraint_pointwise_RMS_N_per_mm2':weak_pressure_rms,
                'max_abs_pressure_N_per_mm2':float(np.max(np.abs(pq))),'matrix_asymmetry_relative':asym,
                'pressure_scale_N_per_mm2':load['factor'],'mixed_DOF':total}
            np.savez_compressed(out/'solution.npz',displacement=u,pressure=p,rhs=rhs,clamped_D=D,
                residual=residual,inner_facets=inner_facets,outer_facets=outer_facets,bottom_free_facets=bottom_facets,
                component0=components[0],component1=components[1],component2=components[2],
                quadrature_strain=strain,quadrature_pressure=pq,quadrature_coordinates_mm=np.asarray(bu.global_coordinates()))
            result={'h':h,'nu_hypothetical':nu,'load_id':load_id,'gates':gates,'passed':all(gates.values()),
                'metrics':metrics,'FEM_rhs_solves':1,'cpu_seconds':time.process_time()-tc,'wall_seconds':time.perf_counter()-ts,
                'material_calibrated':False,'contact_solver':False}
            save(out/'result.json',result);results.append(result)
            if not result['passed']:raise RuntimeError('Local shell gate failed: '+str((h,nu,load_id)))
        del lu
    report={'h':h,'cpu':cpu,'pid':os.getpid(),'mesh_sha256':mesh_sha,'results':results,'FEM_rhs_solves':6,
        'mesh_NPZ_loads':1,'geometry_matrix_assemblies':3,'mixed_factorizations':2,
        'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-ct}
    save(block/'result.json',report)
    return report
