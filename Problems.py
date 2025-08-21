# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:48:29 2024

@author: hp
"""
import numpy as np
import pymoo.problems.many as ppm
def DTLZ1(x,k=1,M=3):
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])#[0.5]#
    g=100*(k+sum([(z[il]-0.5)**2-np.cos(20*np.pi*(z[il]-0.5)) for il in range(k)]))
    f=[0 for il in range(M)]
    f[0]=(1+g)*0.5*np.prod(y[:M-1])
    for fil in range(1,M-1):
        f[fil]=(1+g)*0.5*np.prod(y[:M-fil-1])*(1-y[M-fil-1])
    f[-1]=(1+g)*0.5*(1-y[0])
    return f,0
def DTLZ1_5(x,k=1,M=5):
    f,_=DTLZ1(x,k,M=M)
    return f,0
def DTLZ1optimalresults(M=3):
    n=M
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    if M==3:n_partitions=30
    elif M==5:n_partitions=10
    ref_dirs = get_reference_directions("das-dennis", M, n_partitions=n_partitions)
    fxa = (get_problem("dtlz1").pareto_front(ref_dirs)).tolist()
    x=np.arange(0,25,25/500)
    xa=[[0 for _ in range(M)] for il in range(len(fxa))]
    return xa,fxa
def DTLZ4(x,k=1,M=3,alpha=2):
    n=len(x)
    j=n-k
    y=np.array(x[:j])**alpha
    z=np.array(x[j:])
    g=sum([(z[il]-0.5)**2 for il in range(k)])
    f=[0 for il in range(M)]
    f[0]=(1+g)*np.prod(np.cos(y[:M-1]*np.pi/2))
    for fil in range(1,M-1):
        f[fil]=(1+g)*np.prod(np.cos(y[:M-fil-1]*np.pi/2))*np.sin(y[M-fil-1]*np.pi/2)
    f[-1]=(1+g)*np.sin(y[0]*np.pi/2)
    return f,0
def DTLZ4_5(x,k=1,M=5):
    f,_=DTLZ4(x,k,M=M)
    return f,0
def DTLZ2(x,k=1,M=3):
    f,_=DTLZ4(x,k=1,M=M,alpha=1)
    return f,0
def DTLZ2_5(x,k=1,M=5):
    f,_=DTLZ2(x,k,M=M)
    return f,0
def DTLZ3(x,k=1,M=3,alpha=1):
    n=len(x)
    j=n-k
    y=np.array(x[:j])**alpha
    z=np.array(x[j:])
    g=100*(k+sum([(z[il]-0.5)**2-np.cos(20*np.pi*(z[il]-0.5)) for il in range(k)]))
    f=[0 for il in range(M)]
    f[0]=(1+g)*np.prod(np.cos(y[:M-1]*np.pi/2))
    for fil in range(1,M-1):
        f[fil]=(1+g)*np.prod(np.cos(y[:M-fil-1]*np.pi/2))*np.sin(y[M-fil-1]*np.pi/2)
    f[-1]=(1+g)*np.sin(y[0]*np.pi/2)
    return f,0
def DTLZ3_5(x,k=1,M=5):
    f,_=DTLZ3(x,k=1,M=M)
    return f,0
def DTLZ5(x,k=1,M=3):
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])
    g=sum([(z[il]-0.5)**2 for il in range(k)])
    y[1:M-1]=(1+2*g*y[1:M-1])/(2*(1+g))
    f=[0 for il in range(M)]
    f[0]=(1+g)*np.prod(np.cos(y[:M-1]*np.pi/2))
    for fil in range(1,M-1):
        f[fil]=(1+g)*np.prod(np.cos(y[:M-fil-1]*np.pi/2))*np.sin(y[M-fil-1]*np.pi/2)
    f[-1]=(1+g)*np.sin(y[0]*np.pi/2)
    return f,0
def DTLZ5_5(x,k=1,M=5):
    f,_=DTLZ5(x,k=1,M=M)
    return f,0
def DTLZ6(x,k=1,M=3,alpha=2):
    n=len(x)
    j=n-k
    y=np.array(x[:j])**alpha
    z=np.array(x[j:])
    g=sum([(z[il])**0.1 for il in range(k)])
    y[1:M-1]=(1+2*g*y[1:M-1])/(2*(1+g))
    f=[0 for il in range(M)]
    f[0]=(1+g)*np.prod(np.cos(y[:M-1]*np.pi/2))
    for fil in range(1,M-1):
        f[fil]=(1+g)*np.prod(np.cos(y[:M-fil-1]*np.pi/2))*np.sin(y[M-fil-1]*np.pi/2)
    f[-1]=(1+g)*np.sin(y[0]*np.pi/2)
    return f,0
def DTLZ6_5(x,k=1,M=5):
    f,_=DTLZ6(x,k=1,M=M)
    return f,0
def DTLZ5optimalresults(M=3):
    n=M
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    fxa = (get_problem("dtlz5").pareto_front())
    fxa=(fxa[::20]).tolist()
    x=np.arange(0,25,25/500)
    xa=[[0 for _ in range(M)] for il in range(len(fxa))]
    return xa,fxa
def DTLZ6optimalresults(M=3):
    n=M
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    fxa = (get_problem("dtlz6").pareto_front())
    fxa=(fxa[::20]).tolist()
    x=np.arange(0,25,25/500)
    xa=[[0 for _ in range(M)] for il in range(len(fxa))]
    return xa,fxa
def DTLZ3optimalresults(M=3):
    n=M
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    if M==3:n_partitions=30
    elif M==5:n_partitions=10
    ref_dirs = get_reference_directions("das-dennis", M, n_partitions=n_partitions)
    fxa = (get_problem("dtlz3").pareto_front(ref_dirs)).tolist()
    x=np.arange(0,25,25/500)
    xa=[[0 for _ in range(M)] for il in range(len(fxa))]
    return xa,fxa
def DTLZ2optimalresults(M=3):
    n=M
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    if M==3:n_partitions=30
    elif M==5:n_partitions=10
    ref_dirs = get_reference_directions("das-dennis", M, n_partitions=n_partitions)
    fxa = (get_problem("dtlz2").pareto_front(ref_dirs)).tolist()
    x=np.arange(0,25,25/500)
    xa=[[0 for _ in range(M)] for il in range(len(fxa))]
    return xa,fxa
def DTLZ4optimalresults(M=3):
    n=M
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    if M==3:n_partitions=30
    elif M==5:n_partitions=10
    ref_dirs = get_reference_directions("das-dennis", M, n_partitions=n_partitions)
    fxa = (get_problem("dtlz4").pareto_front(ref_dirs)).tolist()
    x=np.arange(0,25,25/500)
    xa=[[0 for _ in range(M)] for il in range(len(fxa))]
    return xa,fxa
def DTLZ7(x,k=1,M=3):
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])
    g=1+9*sum([z[il]/(k) for il in range(k)])
    f=[0 for il in range(M)]
    for fil in range(M-1):
        f[fil]=y[fil]
    f[-1]=(1+g)*(M-sum([ f[il]/(1+g)*(1+np.sin(3*np.pi*f[il])) for il in range(M-1)]))
    return f,0
def DTLZ7_5(x,k=1,M=5):
    f,_=DTLZ7(x,k=1,M=M)
    return f,0
def DTLZ7optimalresults(M=3):
    n=M
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.visualization.scatter import Scatter
    fxa = (get_problem("dtlz7").pareto_front())
    fxa=(fxa[::20]).tolist()
    x=np.arange(0,25,25/500)
    xa=[[0 for _ in range(M)] for il in range(len(fxa))]
    return xa,fxa
def MMF1(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [1,3] x2 eps [-1,1]
    #global f1=[0,1];f2=1-f1**0.5
    f1=abs(x1-2)
    f2=1-np.sqrt(abs(x1-2))+2*(x2-np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    #np.sin(6*np.pi*abs(x[il]-2)+np.pi)
    f=[f1,f2]
    return f,0
def MMF1optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    x1=np.linspace(1,3,num=n)
    x2=np.sin(6*np.pi*np.abs(x1-2)+np.pi)
    # xa=np.concatenate((x1[np.newaxis].T,x2[np.newaxis].T))
    if draw:
        plt.figure(2)
        plt.plot(x1,x2,'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    for fil in range(n-1):
        f0,_=MMF1([x1[fil],x2[fil]])
        f1,_=MMF1([x1[fil+1],x2[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[-1]=f1
    xa[-1]=[x1[fil+1],x2[fil+1]]
    return xa,fxa
def MMF1e(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [1,3] x2 eps [-1,1]
    #global f1=[0,1];f2=1-f1**0.5
    a=np.e
    f1=abs(x1-2)
    if x1>=1 and x1<2:
        f2=1-np.sqrt(abs(x1-2))+2*(x2-np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    elif x1>=2 and x1<=3:
        f2=1-np.sqrt(abs(x1-2))+2*(x2-a**x1*np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    #np.sin(6*np.pi*abs(x[il]-2)+np.pi)
    f=[f1,f2]
    return f,0
def MMF1eoptimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    a=np.e
    x1=np.linspace(1,3,num=n)
    # if x1>=1 and x1<2:
    #     x2=np.sin(6*np.pi*np.abs(x1-2)+np.pi)
    # elif x1>=2 and x1<=3:
    #     x2=a**x1*np.sin(6*np.pi*np.abs(x1-2)+np.pi)
    x2=[iff(x1[il]>=1 and x1[il]<2,1,a)**x1[il]*np.sin(6*np.pi*np.abs(x1[il]-2)+np.pi)
        for il in range(len(x1))]
    # xa=np.concatenate((x1[np.newaxis].T,x2[np.newaxis].T))
    if draw:
        plt.figure(2)
        plt.plot(x1,x2,'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    for fil in range(n-1):
        f0,_=MMF1e([x1[fil],x2[fil]])
        f1,_=MMF1e([x1[fil+1],x2[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[-1]=f1
    xa[-1]=[x1[fil+1],x2[fil+1]]
    return xa,fxa
def MMF3(x12):
    x1=x12[0]
    x2=x12[1]
    f1=x1
    if (x2>=0 and x2<=0.5) or (x2>0.5 and x2<1 and x1>0.25 and x1<=1):
        f2=1-x1**0.5+2*(4*(x2-x1**0.5)**2-2*np.cos(20*(x2-x1**0.5)*np.pi/2**0.5)+2)
    else:
        f2=1-x1**0.5+2*(4*(x2-0.5-x1**0.5)**2-2*np.cos(20*(x2-0.5-x1**0.5)*np.pi/2**0.5)+2)
    f=[f1,f2]
    return f,0
def MMF2(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [0,1] x2 eps [0,2]
    #global f1=[0,1];f2=1-f1**0.5
    f1=x1
    if 0<=x2 and x2<=1:
        f2=1-x1**0.5+2*(4*(x2-x1**0.5)**2-2*np.cos(20*(x2-x1**0.5)*np.pi/2**0.5)+2)
    elif 1<x2 and x2<=2:
        f2=1-x1**0.5+2*(4*(x2-1-x1**0.5)**2-2*np.cos(20*(x2-1-x1**0.5)*np.pi/2**0.5)+2)
    f=[f1,f2]
    return f,0
def MMF3optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    x2=np.linspace(0,1.5,num=n)
    x1=np.zeros((1,len(x2)))[0]
    x10=[];x20=[]
    cont=-1
    for fil in range(len(x2)):
        x1P=(x2[fil]-0.5)**2
        if (x2[fil]>=0 and x2[fil]<=0.5) or (x2[fil]>0.5 and x2[fil]<1 and x1P>0.25 and x1P<=1):
            cont+=1
            x1P=(x2[fil])**2
            x1[fil]=x1P
            x10.append(x1[fil])
            x20.append(x2[fil])
    
        # plt.figure(2)
    for fil in range(len(x2)): 
        x1P=(x2[fil])**2
        if (x2[fil]>1 and x2[fil]<=1.5) or (x2[fil]>0.5 and x2[fil]<1 and x1P>=0 and x1P<0.25):
            x1P=(x2[fil]-0.5)**2
            x1[fil]=x1P
            x10.append(x1[fil])
            x20.append(x2[fil])
    if draw:            
        plt.figure(2)
        plt.plot(x1,x2,'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    # fxa[-1]=[1,0]
    for fil in range(len(x20)-1):
        f0,_=MMF3([x10[fil],x20[fil]])
        f1,_=MMF3([x10[fil+1],x20[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw and fil!=cont:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[fil+1]=f1
    xa[fil+1]=[x1[fil+1],x2[fil+1]]
    # if draw:plt.plot([f1[0],1],[f1[1],0],color='blue')
    # xa[-1]=[x1[-1],x2[-1]]
    return xa,fxa
def MMF2optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    x2=np.linspace(0,2,num=n)
    x1=np.zeros((1,len(x2)))[0]
    x10=[];x20=[]
    cont=-1
    for fil in range(len(x2)):
        if (x2[fil])>=0 and (x2[fil])<=1:
            cont+=1
            x1[fil]=x2[fil]**2
            x10.append(x1[fil])
            x20.append(x2[fil])
    
        # plt.figure(2)
    for fil in range(len(x2)): 
        if (x2[fil])>1 and (x2[fil])<=2:
            x1[fil]=(x2[fil]-1)**2
            x10.append(x1[fil])
            x20.append(x2[fil])
    if draw:            
        plt.figure(2)
        plt.plot(x1,x2,'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    # fxa[-1]=[1,0]
    for fil in range(len(x20)-1):
        f0,_=MMF2([x10[fil],x20[fil]])
        f1,_=MMF2([x10[fil+1],x20[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw and fil!=cont:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[fil+1]=f1
    xa[fil+1]=[x1[fil+1],x2[fil+1]]
    # if draw:plt.plot([f1[0],1],[f1[1],0],color='blue')
    # xa[-1]=[x1[-1],x2[-1]]
    return xa,fxa
def MMF4(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [-1,1] x2 eps [0,2]
    #global f1=[0,1];f2=1-f1**2
    f1=abs(x1)
    if 0<=x2 and x2<1:
        f2=1-x1**2+2*(x2-np.sin(np.pi*abs(x1)))**2
    elif 1<=x2 and x2<=2:
        f2=1-x1**2+2*(x2-1-np.sin(np.pi*abs(x1)))**2
    f=[f1,f2]
    return f,0
def MMF4optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    # x1=np.linspace(-1,1,num=n)
    x1=np.linspace(-1,1,num=int(n/2));x1=np.concatenate((x1,x1))
    x2=np.zeros((1,len(x1)))[0]
    if draw:
        plt.figure(2)
    for fil in range(int(n/2)):
        x2[fil]=np.sin(np.pi*np.abs(x1[fil]))
    # if draw:
    #     plt.plot(x1[:int(n/2)],x2[:int(n/2)],'blue')
    # x2=np.zeros((1,len(x1)))[0]
    for fil in range(int(n/2),len(x2)):
        x2[fil]=np.sin(np.pi*np.abs(x1[fil]))+1
    if draw:
        plt.plot(x1[:int(n/2)],x2[:int(n/2)],'blue')
        plt.plot(x1[int(n/2):],x2[int(n/2):],'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    for fil in range(n-1):
        f0,_=MMF4([x1[fil],x2[fil]])
        f1,_=MMF4([x1[fil+1],x2[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[-1]=f1
    xa[-1]=[x1[fil+1],x2[fil+1]]
    return xa,fxa
def MMF5(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [-1,3] x2 eps [1,3]
    #global f1=[0,1];f2=1-f1**0.5
    f1=abs(x1-2)
    if -1<=x2 and x2<=1:
        f2=1-(abs(x1-2))**0.5+2*(x2-np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    elif 1<x2 and x2<=3:
        f2=1-(abs(x1-2))**0.5+2*(x2-2-np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    f=[f1,f2]
    return f,0
def MMF5optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    # x1=np.linspace(1,3,num=n)
    x1=np.linspace(1,3,num=int(n/2));x1=np.concatenate((x1,x1))
    x2=np.zeros((1,len(x1)))[0]
    if draw:
        plt.figure(2)
    for fil in range(int(n/2)):
        x2[fil]=np.sin(6*np.pi*np.abs(x1[fil]-2)+np.pi)
    # if draw:plt.plot(x1,x2,'blue')
    # x2=np.zeros((1,len(x1)))[0]
    for fil in range(int(n/2),len(x2)):
        x2[fil]=np.sin(6*np.pi*np.abs(x1[fil]-2)+np.pi)+2
    if draw:
        plt.plot(x1[:int(n/2)],x2[:int(n/2)],'blue')
        plt.plot(x1[int(n/2):],x2[int(n/2):],'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    for fil in range(n-1):
        f0,_=MMF5([x1[fil],x2[fil]])
        f1,_=MMF5([x1[fil+1],x2[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[-1]=f1
    xa[-1]=[x1[fil+1],x2[fil+1]]
    return xa,fxa
def MMF6(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [-1,3] x2 eps [1,2]
    #global f1=[0,1];f2=1-f1**0.5
    f1=abs(x1-2)
    if -1<=x2 and x2<=1:
        f2=1-(abs(x1-2))**0.5+2*(x2-np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    elif 1<x2 and x2<=2:
        f2=1-(abs(x1-2))**0.5+2*(x2-1-np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    f=[f1,f2]
    return f,0
def MMF6optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    # x1=np.linspace(1,3,num=n)
    x1=np.linspace(1,3,num=int(n/2));x1=np.concatenate((x1,x1))
    x2=np.zeros((1,len(x1)))[0]
    if draw:plt.figure(2)
    for fil in range(int(n/2)):
        x2[fil]=np.sin(6*np.pi*np.abs(x1[fil]-2)+np.pi)
    # if draw:plt.plot(x1,x20,'blue')
    # x2=np.zeros((1,len(x1)))[0]
    for fil in range(int(n/2),len(x2)):
        x2[fil]=np.sin(6*np.pi*np.abs(x1[fil]-2)+np.pi)+1
    if draw:
        plt.plot(x1[:int(n/2)],x2[:int(n/2)],'blue')
        plt.plot(x1[int(n/2):],x2[int(n/2):],'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    for fil in range(n-1):
        # f0,_=MMF6([x1[fil],x2[fil]])
        # f1,_=MMF6([x1[fil+1],x2[fil+1]])
        f0=[abs(x1[fil]-2),1-abs(x1[fil]-2)**0.5]
        f1=[abs(x1[fil+1]-2),1-abs(x1[fil+1]-2)**0.5]
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[-1]=f1
    xa[-1]=[x1[fil+1],x2[fil+1]]
    return xa,fxa
def MMF7(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [-1,3] x2 eps [1,3]
    #global f1=[0,1];f2=1-f1**0.5
    f1=abs(x1-2)
    f2=1-(abs(x1-2))**0.5+(x2-(0.3*abs(x1-2)**2*np.cos(24*np.pi*abs(x1-2)+4*np.pi)+0.6*abs(x1-2))*np.sin(6*np.pi*abs(x1-2)+np.pi))**2
    f=[f1,f2]
    return f,0
def MMF7optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    x1=np.linspace(1,3,num=n)
    x2=(0.3*np.abs(x1-2)**2*np.cos(24*np.pi*np.abs(x1-2)+4*np.pi)+0.6*np.abs(x1-2))*np.sin(6*np.pi*np.abs(x1-2)+np.pi)
    if draw:
        plt.figure(2)
        plt.plot(x1,x2,'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    for fil in range(n-1):
        f0,_=MMF7([x1[fil],x2[fil]])
        f1,_=MMF7([x1[fil+1],x2[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[-1]=f1
    xa[-1]=[x1[fil+1],x2[fil+1]]
    return xa,fxa
def MMF8(x12):
    x1=x12[0]
    x2=x12[1]
    #x1 eps [-np.pi,np.pi] x2 eps [0,9]
    #global f1=[0,1];f2=1-f1**0.5
    f1=np.sin(abs(x1))
    if 0<=x2 and x2<=4:
        f2=(1-np.sin(abs(x1))**2)**0.5+2*(x2-np.sin(np.abs(x1))-abs(x1))**2
    elif 4<x2 and x2<=9:
        f2=(1-np.sin(abs(x1))**2)**0.5+2*(x2-4-np.sin(np.abs(x1))-abs(x1))**2
    f=[f1,f2]
    return f,0
def MMF8optimalresults(n=500,draw=True):
    import matplotlib.pyplot as plt
    # x1=np.linspace(-np.pi,np.pi,num=n)
    x1=np.linspace(-np.pi,np.pi,num=int(n/2));x1=np.concatenate((x1,x1))
    x2=np.zeros((1,len(x1)))[0]
    if draw:plt.figure(2)
    for fil in range(int(n/2)):
        x2[fil]=np.sin(np.abs(x1[fil]))+np.abs(x1[fil])
    # if draw:plt.plot(x1,x20,'blue')
    # x2=np.zeros((1,len(x1)))[0]
    for fil in range(int(n/2),len(x2)):
        x2[fil]=np.sin(np.abs(x1[fil]))+np.abs(x1[fil])+4
    if draw:
        plt.plot(x1[:int(n/2)],x2[:int(n/2)],'blue')
        plt.plot(x1[int(n/2):],x2[int(n/2):],'blue')
        plt.figure(3)
    fxa=[[0,0] for il in range(n)]
    xa=[[0,0] for il in range(n)]
    for fil in range(n-1):
        f0,_=MMF8([x1[fil],x2[fil]])
        f1,_=MMF8([x1[fil+1],x2[fil+1]])
        fxa[fil]=f0
        xa[fil]=[x1[fil],x2[fil]]
        if draw:plt.plot([f0[0],f1[0]],[f0[1],f1[1]],color='blue')
    fxa[-1]=f1
    xa[-1]=[x1[fil+1],x2[fil+1]]
    return xa,fxa
def ZDT1(x,k=1):
    #Domain x=[0,1]
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])#[0.5]#
    g=(1+9*sum([z[il]/k for il in range(k)]))
    f1=y[0]
    h=1-(f1/g)**0.5
    f=[f1,g*h]
    return f,0
def ZDT1optimalresults(draw=True):
    importinstall('pymoo')
    from pymoo.problems import get_problem
    n_partitions=30
    fxa = (get_problem("zdt1").pareto_front()).tolist()
    if draw:
        import matplotlib.pyplot as plt
        plt.figure(3)
        for fil in range(len(fxa)-1):
            plt.plot([fxa[fil][0],fxa[fil+1][0]],[fxa[fil][1],fxa[fil+1][1]],color='blue')
    xa=[[0 for _ in range(2)] for il in range(len(fxa))]
    # fxa,_=DTLZ2(xa,M=M,k=k)
    return xa,fxa
def ZDT2(x,k=1):
    #Domain x=[0,1]
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])#[0.5]#
    g=(1+9*sum([z[il]/k for il in range(k)]))
    f1=y[0]
    h=1-(f1/g)**2
    f=[f1,g*h]
    return f,0
def ZDT2optimalresults(draw=True):
    importinstall('pymoo')
    from pymoo.problems import get_problem
    n_partitions=30
    fxa = (get_problem("zdt2").pareto_front()).tolist()
    if draw:
        import matplotlib.pyplot as plt
        plt.figure(3)
        for fil in range(len(fxa)-1):
            plt.plot([fxa[fil][0],fxa[fil+1][0]],[fxa[fil][1],fxa[fil+1][1]],color='blue')
    xa=[[0 for _ in range(2)] for il in range(len(fxa))]
    # fxa,_=DTLZ2(xa,M=M,k=k)
    return xa,fxa
def ZDT3(x,k=1):
    #Domain x=[0,1]
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])#[0.5]#
    g=(1+9*sum([z[il]/k for il in range(k)]))
    f1=y[0]
    h=1-(f1/g)**0.5-(f1/g)*np.sin(10*np.pi*f1)
    f=[f1,g*h]
    return f,0
def ZDT3optimalresults(draw=True):
    importinstall('pymoo')
    from pymoo.problems import get_problem
    n_partitions=30
    fxa = (get_problem("zdt3").pareto_front()).tolist()
    if draw:
        import matplotlib.pyplot as plt
        plt.figure(3)
        for fil in range(len(fxa)-1):
            plt.plot([fxa[fil][0],fxa[fil+1][0]],[fxa[fil][1],fxa[fil+1][1]],color='blue')
    xa=[[0 for _ in range(2)] for il in range(len(fxa))]
    # fxa,_=DTLZ2(xa,M=M,k=k)
    return xa,fxa
def ZDT4(x,k=1):
    #Domain x=[0,1]
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])#[0.5]#
    g=(1+10*k+sum([z[il]**2-10*np.cos(4*np.pi*z[il]) for il in range(k)]))
    f1=y[0]
    h=1-(f1/g)**0.5
    f=[f1,g*h]
    return f,0
def ZDT4optimalresults(draw=True):
    importinstall('pymoo')
    from pymoo.problems import get_problem
    n_partitions=30
    fxa = (get_problem("zdt4").pareto_front()).tolist()
    if draw:
        import matplotlib.pyplot as plt
        plt.figure(3)
        for fil in range(len(fxa)-1):
            plt.plot([fxa[fil][0],fxa[fil+1][0]],[fxa[fil][1],fxa[fil+1][1]],color='blue')
    xa=[[0 for _ in range(2)] for il in range(len(fxa))]
    # fxa,_=DTLZ2(xa,M=M,k=k)
    return xa,fxa
def fScalInv(Pob_float,lim,ne,LC,Nr):
    Pob_10=[0 for _ in range(Nr*ne)]
    for i in range(Nr):
        for j in range(ne):
            r=(i)*ne+j
            if (lim[j][1]-lim[j][0])!=0:
                Pob_10[r]=int(round((Pob_float[r]-lim[j][0])*(2**LC-1)/(lim[j][1]-lim[j][0]),0))
    return Pob_10
def fPobDecInv(Pob_10,ne,Nr,LC):
    Pob_2=np.zeros((Nr*ne,LC)).tolist()
    Pob_2N=[bin(Pob_10[i]) for i in range(len(Pob_10))]
    for i in range(Nr*ne):
        bi=Pob_2N[i]
        cont=LC
        for j in range(len(bi)-1,2-1,-1):
            cont-=1
            Pob_2[i][cont]=int(bi[j])
    return Pob_2
def ZDT5(x,k=1):
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])
    LC=30
    ne=len(y)
    lim=[[0,1,'float'] for _ in range(ne)]
    Nr=1
    Pob_float=y
    Pob_10=fScalInv(Pob_float,lim,ne,LC,Nr)
    Pob_2=fPobDecInv(Pob_10,ne,Nr,LC)
    uy=[sum(Pob_2[il]) for il in range(len(Pob_2))]
    LC=5
    ne=len(z)
    lim=[[0,1,'float'] for _ in range(ne)]
    Nr=1
    Pob_float=z
    Pob_10=fScalInv(Pob_float,lim,ne,LC,Nr)
    Pob_2=fPobDecInv(Pob_10,ne,Nr,LC)
    uz=[sum(Pob_2[il]) for il in range(len(Pob_2))]
    v=[2+uz[il] if uz[il]<5 else 1 for il in range(len(uz))]
    g=sum([v[il] for il in range(k)])
    f1=1+uy[0]
    h=1/f1
    f=[f1,g*h]
    return f,0
def ZDT5optimalresults(draw=True):
    importinstall('pymoo')
    from pymoo.problems import get_problem
    n_partitions=30
    fxa = (get_problem("zdt5",2,2,normalize=False).pareto_front()).tolist()
    if draw:
        import matplotlib.pyplot as plt
        plt.figure(3)
        for fil in range(len(fxa)-1):
            plt.plot([fxa[fil][0],fxa[fil+1][0]],[fxa[fil][1],fxa[fil+1][1]],color='blue')
    xa=[[0 for _ in range(2)] for il in range(len(fxa))]
    return xa,fxa
def ZDT6(x,k=1):
    n=len(x)
    j=n-k
    y=np.array(x[:j])
    z=np.array(x[j:])#[0.5]#
    g=(1+9*sum([z[il]/k for il in range(k)])**0.25)
    f1=1-np.exp(-4*y[0])*np.sin(6*np.pi*y[0])**6
    h=1-(f1/g)**2
    f=[f1,g*h]
    return f,0
def ZDT6optimalresults(draw=True):
    importinstall('pymoo')
    from pymoo.problems import get_problem
    # from pymoo.util.plotting import plot
    n_partitions=30
    fxa = (get_problem("zdt6").pareto_front()).tolist()
    if draw:
        import matplotlib.pyplot as plt
        plt.figure(3)
        for fil in range(len(fxa)-1):
            plt.plot([fxa[fil][0],fxa[fil+1][0]],[fxa[fil][1],fxa[fil+1][1]],color='blue')
    xa=[[0 for _ in range(2)] for il in range(len(fxa))]
    return xa,fxa
def s_linear(y,A):
    return np.abs(y-A)/np.abs(np.floor(A-y)+A)
def b_flat(y,A,B,C):
    Output=A+min(0,floor(y-B))*A*(B-y)/B-min(0,floor(C-y))*(1-A)*(y-C)/(1-C)
    return round(Output*1e4)/1e4
def b_poly(y,a):
    return y**a
def r_sum(y,w):
    return sum([y[il]*w[il] for il in range(len(w))])/sum(w)
def convex(x):
    cos_part = 1 - np.cos(x[:, :-1] * np.pi / 2)
    sin_part = 1 - np.sin(x[:, ::-1] * np.pi / 2) 
    return np.cumprod(np.c_[np.ones_like(x[:, :1]), cos_part], axis=1) * np.c_[np.ones_like(x[:, :1]), sin_part]
def mixed(x):
    return 1-x[:,0]-np.cos(10*np.pi*x[:,0]+np.pi/2)/10/np.pi
# def WFG1(x,M=3):
    # wfg = ppm.wfg.WFG1(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    # if min(wfg)<0:wfg[np.argmin(wfg)]=100
    # return wfg,0
# def WFG1optimalresults(D=10,M=3):
    # wfg=ppm.wfg.WFG1(n_var=D, n_obj=M)
    # fxa = (wfg.pareto_front()).tolist()
    # xa=(wfg.pareto_set()).tolist()
    # return xa,fxa
def WFG2(x,M=3):
    wfg = ppm.wfg.WFG2(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG2optimalresults(D=10,M=3):
    wfg=ppm.wfg.WFG2(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa
def WFG3(x,M=3):
    wfg = ppm.wfg.WFG3(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG3optimalresults(D=10,M=3):
    wfg=ppm.wfg.WFG3(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa
def WFG4(x,M=3):
    wfg = ppm.wfg.WFG4(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG4optimalresults(D=10,M=3):
    wfg=ppm.wfg.WFG4(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa
def WFG5(x,M=3):
    wfg = ppm.wfg.WFG5(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG5optimalresults(D=10,M=3):
    wfg=ppm.wfg.WFG5(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa
def WFG6(x,M=3):
    wfg = ppm.wfg.WFG6(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG6optimalresults(D=10,M=3):
    wfg=ppm.wfg.WFG6(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa
def WFG7(x,M=3):
    wfg = ppm.wfg.WFG7(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG7optimalresults(D=10,M=3):
    wfg=ppm.wfg.WFG7(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa
def WFG8(x,M=3):
    wfg = ppm.wfg.WFG8(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG8optimalresults(D=10,M=3):
    wfg=ppm.wfg.WFG8(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa
def WFG9(x,M=3):
    wfg = ppm.wfg.WFG9(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    if min(wfg)<0:wfg[np.argmin(wfg)]=100
    return wfg,0
def WFG1(x,M=3):
    wfg = ppm.wfg.WFG1(n_var=len(x), n_obj=M).evaluate([x])[0].tolist()#, k=K
    return wfg,0
def WFG1optimalresults(D=10,M=3):
    importinstall('pymoo')
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    n_partitions=30
    wfg=ppm.wfg.WFG1(n_var=D, n_obj=M)
    fxa = (wfg.pareto_front()).tolist()
    xa=(wfg.pareto_set()).tolist()
    return xa,fxa