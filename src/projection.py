import numpy as np

def project(q):
    x,y,z=q
    angle=np.arctan2(y,x)
    side=np.array([20*np.cos(angle),20*np.sin(angle),np.clip(z,0,100)])
    d=np.array([x,y,max(z-100,0)])
    if np.linalg.norm(d)==0:
        return side
    cap=np.array([0,0,100])+20*d/np.linalg.norm(d)
    return min((side,cap),key=lambda p:np.linalg.norm(p-q))

