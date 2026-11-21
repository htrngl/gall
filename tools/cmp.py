import sys,os,numpy as np
a,b=sys.argv[1:3]
def cmpf(x,y,dt):
    u=np.fromfile(x,dt).astype(np.float64); v=np.fromfile(y,dt).astype(np.float64)
    n=min(len(u),len(v)); u,v=u[:n],v[:n]
    return np.abs(u-v).max() if n else 0, np.abs(u).max() if n else 0
if os.path.isdir(a):
    w=(0,0,'');nb=0;fs=[f for f in os.listdir(a) if f!='index.tsv']
    for f in fs:
        if open(a+'/'+f,'rb').read()==open(b+'/'+f,'rb').read(): nb+=1; continue
        d,m=cmpf(a+'/'+f,b+'/'+f,{'f32':np.float32,'i32':np.int32,'f16':np.float16}[f.rsplit('.',1)[1]])
        if d/(m+1e-9)>w[0]: w=(d/(m+1e-9),d,f)
    print(a,b,len(fs),'files',nb,'bitwise equal; worst rel',w)
else:
    V=200058
    u=np.fromfile(a,np.float32).reshape(-1,V); v=np.fromfile(b,np.float32).reshape(-1,V)
    n=min(len(u),len(v)); d=np.abs(u[:n]-v[:n]).max(1)
    print(a,b,n,'rows; max abs diff per row: max %.3g mean %.3g'%(d.max(),d.mean()),'argmax agree',(u[:n].argmax(1)==v[:n].argmax(1)).sum(),'/',n, 'first rows',d[:4])
