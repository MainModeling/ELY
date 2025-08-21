# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:41:38 2024

@author: hp
"""
import numpy as np
import scipy.stats as stats
import scipy as sp
import importlib
import copy
import sys
import time
#%% ELY
def fPobDec(Pob_2,ne,Nr):
    Pob_10=[0 for _ in range(Nr*ne)]
    Pob_2N=[]
    for i in range(Nr*ne):
        Pob_2Ni='0b'
        for j in range(len(Pob_2[0])):
            Pob_2Ni=Pob_2Ni+str(Pob_2[i][j])
        Pob_2N.append(Pob_2Ni)
        
    for i in range(Nr*ne):
        Pob_10[i]=int(Pob_2N[i],2)
    return Pob_10
def fScale(Pob_10,lim,ne,LC,Nr,ndecimals=4):
    Pob_float=[0 for _ in range(Nr*ne)]
    for i in range(Nr):
        for j in range(ne):
            r=(i)*ne+j
            if lim[j][2]=='int':
                Pob_float[r]=int(round(Pob_10[r]/(2**LC-1)*(lim[j][1]-lim[j][0])+lim[j][0],0))
            elif lim[j][2]=='float':
                Pob_float[r]=float(round(Pob_10[r]/(2**LC-1)*(lim[j][1]*(10**ndecimals)-lim[j][0]*(10**ndecimals))+lim[j][0]*(10**ndecimals),0)/(10**ndecimals))
    return Pob_float
def fScalInv(Pob_float,lim,ne,LC,Nr):
    Pob_10=[0 for _ in range(Nr*ne)]
    for i in range(Nr):
        for j in range(ne):
            r=(i)*ne+j
            if (lim[j][1]-lim[j][0])!=0:
                Pob_10[r]=int(round((Pob_float[r]-lim[j][0])*(2**LC-1)/(lim[j][1]-lim[j][0]),0))
    return Pob_10
def fPobDecInv(Pob_10,ne,Nr,LC):
    Pob_2=np.zeros((Nr*ne,LC),dtype=int).tolist()
    Pob_2N=[bin(Pob_10[i]) for i in range(len(Pob_10))]
    for i in range(Nr*ne):
        bi=Pob_2N[i]
        cont=LC
        for j in range(len(bi)-1,2-1,-1):
            cont-=1
            Pob_2[i][cont]=int(bi[j])
    return Pob_2
def SBX(nc=2):
    eta=np.random.rand()
    if eta<=0.5:
        beta=(2*eta)**(1/(nc+1))
    else:
        beta=(2*(1-eta))**(1/(nc+1))
    C11=0.5*(1-beta)
    pcross=C11
    return pcross
def Distance_parameter_filter(Pob_floatt,ne,contG,lim,Pob_floatc,Nrt,Nrc,indcv,limCV=0.01,epsilon=1e-10):

    Pob_floatcM=np.array([Pob_floatc[ne*il:(il+1)*ne] for il in range(Nrc)])
    
    epss=epsilon
    maxpop=-np.array([lim[il][0] for il in range(ne)])
    avg=np.average(Pob_floatcM+maxpop+epss,axis=0)
    std=np.std(Pob_floatcM+maxpop+epss,axis=0)
    CV=std/avg
    
    for fil in range(ne):
        if CV[fil]<limCV:
            avgg=avg[fil]-(maxpop[fil]+epss)/Nrc
            limi=(avgg-std[fil]*3)-epss
            lims=(avgg+std[fil]*3)+epss

            for fill in range(Nrt):
                if max(lim[fil][0],Pob_floatt[fill*ne+fil]-epss)<=limi or min(lim[fil][1],Pob_floatt[fill*ne+fil]+epss)>=lims:
                    indcv[fill]=False
    return np.arange(Nrt)[indcv],lim
def approximate_non_dominated_sorting(indcv,lim,ne,fxt,multiobjectives,epsilon,Nrelite,Pob_floatt):
    randmix=np.random.permutation(indcv);Nrt=len(indcv)
    fxt=[fxt[il] for il in randmix]
    Pob_floatt=[Pob_floatt[il*ne+iil] for il in randmix for iil in range(ne)]
    front=[[],[]];front0a=[];Pcontreal=0
    Nrts=[int(Nrt/multiobjectives)*il for il in range(multiobjectives)]+[Nrt]
    Nrelites=[int(Nrelite/multiobjectives)]*(multiobjectives-1)+[Nrelite-int(Nrelite/multiobjectives)*(multiobjectives-1)]
    Pcontreal=0
    frontsave_=[]
    for objused in range(multiobjectives):
        #Algorithm3
        Pcont,Pcont1,pareto_front,front0,front1,objused=basic_elite(\
            Nrelites[objused],Nrts[objused+1],multiobjectives,objused,fxt[Nrts[objused]:Nrts[objused+1]],epsilon,vectorization=False,traditional=False)
        
        Pcontreal_=Nrelites[objused]-1;front0a=[]
        if (Pcont>Nrelites[objused]) or Pcont>Nrts[objused+1]*0.8:
            #Algorithm4
            Pcont,Pcont1,pareto_front,front0,front1,objused,Pcontreal_,front0a=filter_elite(\
                Pcont,Pcont1,Nrelites[objused],Nrts[objused+1],multiobjectives,pareto_front,front0,front1,objused,Pcontreal_,front0a)
    
        front0_=[front0[:Pcont].tolist()]
        front1_=[front1[:Pcont1].tolist()]
        front_=front0_+front1_
        
        if Nrelites[objused] <= Pcontreal_:
            #Algorithm5
            crowdingdistance=m_crowding_distance(multiobjectives,np.array(fxt)[Nrts[objused]:Nrts[objused+1]][front_[0]],objused)
            front0crowdsorted=np.array(front_[0])[np.argsort(crowdingdistance)[::-1]].tolist()
    
            front_[0]=front0crowdsorted[:Nrelites[objused]]
            frontsave_+=(np.array(front0crowdsorted[Nrelites[objused]:],dtype=int)+Nrts[objused]).tolist()
        
        front[0]+=(np.array(front_[0],dtype=int)+Nrts[objused]).tolist()
        front[1]+=(np.array(front_[1],dtype=int)+Nrts[objused]).tolist()
    front[0]+=frontsave_
    return front,Pcontreal,Pob_floatt,fxt
def basic_elite(Nr,Nrt,multiobjectives,objused,fx,epsilon,vectorization=False,traditional=False):
    fx=np.round(fx,int(str(epsilon)[3:]))
    pareto_front=np.zeros((Nrt,multiobjectives))
    pareto_front1=np.zeros((Nrt,multiobjectives))
    front0=np.zeros((1,Nrt),dtype=int)[0]
    front1=np.zeros((1,Nrt),dtype=int)[0]
    Pcont=0;Pcont1=0
    if objused==multiobjectives:objused=0
    sortedM=np.argsort(fx[:,objused])
    if vectorization==True:
        pass
    else:
        Pcont+=1
        j=sortedM[0]
        pareto_front[Pcont-1]=np.copy(fx[j])
        front0[Pcont-1]=j
        for j in sortedM[1:]:
            point=fx[j].copy()
            if traditional==False:dominated=False
            else:dominated=True
            dominate=False
            jj=Pcont-1
            dominated1=0;dominated2=True;
            dominate1=0
            for k in range(multiobjectives):
                if pareto_front[jj,k]*(1-epsilon)<=point[k]:
                    dominated1+=1
            if traditional and dominate1==multiobjectives and not (point==pareto_front[jj,:]).all():
                dominate=True
            if dominated1==multiobjectives:
                if dominated2:
                    dominated=True
            if (dominated==False or dominate==True):
                    Pcont+=1
                    pareto_front[Pcont-1]=point
                    front0[Pcont-1]=j
            else:
                Pcont1+=1
                front1[Pcont1-1]=j
                pareto_front1[Pcont1-1]=point
    return Pcont,Pcont1,pareto_front,front0,front1,objused
def filter_elite(Pcont,Pcont1,Nr,Nrt,multiobjectives,pareto_front,front0,front1,objused,Pcontreal,front0a):
    contcontrol=0
    Pconta=Pcont;Pcont1a=Pcont1
    cne=np.delete(np.arange(multiobjectives),objused)
    verif=np.random.choice(cne,1)[0]
    if verif==multiobjectives:verif=0
    for multo in [verif]:
        contcontrol+=1
        Pcont2=Pconta
        front00=np.copy(front0);front1a=np.copy(front1)
        front0a=np.zeros((1,Pconta),dtype=int)[0]
        sortedM2=np.argsort(pareto_front[:Pcont2,multo])
        Pconta=0
        Pconta+=1
        pareto_front2=np.zeros((Pcont2,multiobjectives))
        j=sortedM2[0]
        pareto_front2[Pconta-1,:]=np.copy(pareto_front[j,:]);front0a[Pconta-1]=front00[j]
        for j in sortedM2[1:]:
            point=pareto_front[j]
            if not ((pareto_front2[Pconta-1,:]<=point).all() and (pareto_front2[Pconta-1,:]<point).any()):
                Pconta+=1
                pareto_front2[Pconta-1]=point
                front0a[Pconta-1]=front00[j]
            else:
                Pcont1a+=1
                front1a[Pcont1a-1]=front00[j]
        pareto_front=np.copy(pareto_front2)
    Pcontreal=Pconta
    if Pcont>=Nr or Pcont>Nrt*0.8:#False:#
        front0=front0a
        front1=front1a
        Pcont=Pconta
        # objused+=1
    front0a=front0a[:Pconta].tolist()
    return Pcont,Pcont1,pareto_front,front0,front1,objused,Pcontreal,front0a
def m_crowding_distance(multiobjectives,fx,cmultiobjectives):
    
    current_index = 0
    distance=0
    y=np.copy(fx)[:,cmultiobjectives]
    i=cmultiobjectives
    index_of_objectives=np.argsort(y)
    sorted_based_on_objective=y[index_of_objectives]
    y[index_of_objectives[-1]]=np.Inf
    y[index_of_objectives[0]]=np.Inf
    for j in range(1,len(index_of_objectives)-1):
        next_obj=sorted_based_on_objective[j+1]
        previous_obj=sorted_based_on_objective[j-1]
        y[index_of_objectives[j]]=(next_obj-previous_obj)
    distance=np.zeros((len(fx),1))
    i =cmultiobjectives
    crowdingdistance=y
    return crowdingdistance
def Outlier_filter(front0,front1,Nr,Nrt,multiobjectives,pareto_front,significancesl=1e-2):
    Pcontreal=Nr-1;front0a=[]
    Pcont=len(front0);Pcont1=len(front1)
    front0=np.array(front0); front1=np.array(front1)

    contcontrol=0
    Pconta=Pcont;Pcont1a=Pcont1
    nnn=Pconta
    front1a=np.zeros((1,Nrt),dtype=int)[0]
    front1a[:Pcont1]=front1
    for multo in range(multiobjectives):
        C5=-1
        C1=sum(nnn*np.array(pareto_front[:,multo])**2);C2=nnn*(nnn-1);C3=sum(np.array(pareto_front[:,multo]))
        tt = stats.t.ppf(q=1-significancesl/nnn,df=nnn-2)
        Tlim=(nnn-1)/nnn**0.5*(tt**2/(nnn-2+tt**2))**0.5
        C4=C2*C3 + C3*Tlim**2*nnn**2 - C2*C3*nnn
        C5=((nnn*C3**2 + C1 - C1*nnn)*(Tlim**2*nnn**2 - C2*nnn + C2))
        C6=(-Tlim**2*nnn**3 + Tlim**2*nnn**2 + C2*nnn**2 - 2*C2*nnn + C2)
        if C5>=0:
            blimR=-(C4 - Tlim*nnn*C5**(1/2))/C6
            blimL=-(C4 + Tlim*nnn*C5**(1/2))/C6
        else:
            blimR=np.inf
            blimL=-np.inf
        bR=np.where(pareto_front[:,multo]>=blimR)[0]
        bL=np.where(pareto_front[:,multo]<=blimL)[0]
        if len(bR)>0 and len(bL)>0:
            bRbL=np.concatenate((bR,bL))
            NbRbL=np.ones((1,Pcont),dtype=bool)[0]
            NbRbL[np.arange(Pcont)[bRbL]]=False
            pareto_front=pareto_front[NbRbL]
            front1a[Pcont1:Pcont1+len(bRbL)]=front0[bRbL]
            Pcont1+=len(bRbL)
            front0=front0[NbRbL]
            Pcont=len(pareto_front)
        elif len(bR)>0:
            NbR=np.ones((1,Pcont),dtype=bool)[0]
            NbR[np.arange(Pcont)[bR]]=False
            pareto_front=pareto_front[NbR]
            front1a[Pcont1:Pcont1+len(bR)]=front0[bR]
            Pcont1+=len(bR)
            front0=front0[NbR]
            Pcont=len(pareto_front)
        elif len(bL)>0:
            NbL=np.ones((1,Pcont),dtype=bool)[0]
            NbL[np.arange(Pcont)[bL]]=False
            pareto_front=pareto_front[NbL]
            front1a[Pcont1:Pcont1+len(bL)]=front0[bL]
            Pcont1+=len(bL)
            front0=front0[NbL]
            Pcont=len(pareto_front)
    front1=front1a[:Pcont1].tolist()
    front0=front0.tolist()
    Pcontreal=Pcont
        
    return front0,front1,Pcontreal
def Crossover_mutationELY(Pob_float,Nr,ne,LC,MutationPorc,lim,SpecialCrossoverOperator='SBX'):
    Pob_10=fScalInv(Pob_float,lim,ne,LC,Nr)
    Pob_2=fPobDecInv(Pob_10,ne,Nr,LC)
    Parents=np.random.permutation(Nr)
    Pob_C=[0 for _ in range(len(Pob_2))]
    tt=int(Nr/2)
    # print(MutationPorc)
    for t in range(tt):
        if np.random.rand()<MutationPorc/100/2:
            r=(Parents[t])*ne
            Pob_2[r+np.random.randint(ne)][np.random.randint(0,LC)]*=-1+1
            Pob_C[t*ne:(t+1)*ne]=Pob_2[r:r+ne]
            r1=(Parents[tt+t])*ne
            Pob_2[r1+np.random.randint(ne)][np.random.randint(0,LC)]*=-1+1
            Pob_C[int(len(Pob_2)/2)+t*ne:int(len(Pob_2)/2)+(t+1)*ne]=Pob_2[r1:r1+ne]
        else:
            for i in range(ne):
                if SpecialCrossoverOperator=='SBX':
                    pcross=int(SBX(nc=2)*LC)
                else:
                    pcross=np.random.randint(0,LC)
                r=(Parents[t])*ne+i
                aux1=Pob_2[r][0:pcross]
                aux2=Pob_2[r][pcross:LC]
                r1=(Parents[tt+t])*ne+i
                aux3=Pob_2[r1][0:pcross]
                aux4=Pob_2[r1][pcross:LC]
                Pob_C[t*ne+i]=aux1+aux4
                Pob_C[int(len(Pob_2)/2)+t*ne+i]=aux3+aux2
    Pob_10= fPobDec(Pob_C,ne,Nr)
    Pob_float = fScale(Pob_10,lim,ne,LC,Nr)
    return Pob_float,Parents,0
def Crossover_mutationELY2(Pob_float,Nr,ne,LC,MutationPorc,lim,nc=2):
    Parents=np.random.permutation(Nr)
    Pob_C=[0 for _ in range(len(Pob_float))]
    tt=int(Nr/2)

    for t in range(tt):
        if np.random.rand()>=MutationPorc/100/2:
            eta=np.random.rand()
            if eta<=0.5:
                beta=(2*eta)**(1/(nc+1))
            else:
                beta=(1/(2*(1-eta)))**(1/(nc+1))
            r=(Parents[t])*ne
            r1=(Parents[tt+t])*ne
            for fil in range(ne):
                Pob_C[t*ne+fil]=min(lim[fil][1],max(lim[fil][0],0.5*(((1 + beta)*Pob_float[r+fil]) + (1 - beta)*Pob_float[r1+fil])))
                # Pob_C[t*ne+fil]=(Pob_float[r+fil]+Pob_float[r1+fil])/2+0.5*beta*(Pob_float[r+fil]-Pob_float[r1+fil])
                # Pob_C[t*ne+fil]=eta*Pob_float[r+fil]+(1-eta)*Pob_float[r1+fil]
                Pob_C[int(len(Pob_float)/2)+t*ne+fil]=min(lim[fil][1],max(lim[fil][0],0.5*(((1 - beta)*Pob_float[r+fil]) + (1 + beta)*Pob_float[r1+fil])))
                # Pob_C[int(len(Pob_float)/2)+t*ne+fil]=(Pob_float[r+fil]+Pob_float[r1+fil])/2-0.5*beta*(Pob_float[r+fil]-Pob_float[r1+fil])
                # Pob_C[int(len(Pob_float)/2)+t*ne+fil]=eta*Pob_float[r1+fil]+(1-eta)*Pob_float[r+fil]
                # print(Pob_C[t*ne+fil])
        else:
            r=(Parents[t])*ne
            r1=(Parents[tt+t])*ne
            for fil in range(ne):
 
                Pob_C[t*ne+fil]=Pob_float[r+fil]+0.1*np.random.rand()*(lim[fil][1]-Pob_float[r+fil])
                Pob_C[int(len(Pob_float)/2)+t*ne+fil]=Pob_float[r1+fil]-0.1*np.random.rand()*(Pob_float[r1+fil]-lim[fil][0])

    return Pob_C,Parents,0
def Objects(Nrt,Nrc,ne,multiobjectives,indc,fxt,Pob_floatt):
    Pob_floatc=np.zeros((1,Nrc*ne))[0].tolist()
    fxc=np.zeros((Nrc,multiobjectives)).tolist()
    fil=0
    fil2=0
    while fil2<len(indc) and fil<Nrc:
        fxc[fil]=fxt[indc[fil2]]
        Pob_floatc[fil*ne:(fil+1)*ne]=Pob_floatt[indc[fil2]*ne:(indc[fil2]+1)*ne]
        fil+=1
        fil2+=1
    return fxc,Pob_floatc
def ELY(functionanalysis,lim,NumIndiv,MutationPorc,NGenerations,Nstopelite=10,multiobjectives=3,\
        LengthChromosome=0,ndecimals4float=10,printcont=1,epsilon_bar=1e-10,limCV=0.2,binary=False):
    #Objective function
    functionanalysis.index('.');analysismodule='';aux=0;function=''
    for i in range(len(str(functionanalysis))):
        if functionanalysis[i]=='.':aux=1
        if aux==0:
            analysismodule=analysismodule+functionanalysis[i]
        elif functionanalysis[i]!='.':
           function=function+functionanalysis[i]
    if function!='py':
        Module = importlib.import_module(analysismodule)
        method=getattr(Module,function)
        
    def ObjSpace(Pob_float):
        results=[]
        Nr11=int(len(Pob_float)/ne)
        resultsi=[]
        breaking=False
        for i in range(Nr11):
            ri=(i)*ne
            rj=(i+1)*ne
            x=Pob_float[ri:rj]
            resultsi=method(x)
            if resultsi[0]==0:
                breaking=True
            results.append(resultsi)
        fx=[results[il][0] for il in range(len(results))]
        return fx,breaking
        
    #Initial random population
    Nr=NumIndiv
    lim=[copy.deepcopy(lim[i]) for i in range(len(lim))]
    
    intfloat=[lim[i][2] for i in range(len(lim))]
    if LengthChromosome==0 and intfloat.__contains__('float'):LengthChromosome=40
    LCmin=max(5,len(bin(int(round(max([lim[i][1]-lim[i][0] for i in range(len(lim))]),0)+1)))-1)
    if LengthChromosome==[] or LengthChromosome=='':LengthChromosome=0
    if LCmin>LengthChromosome:print('LengthChromosome less than LengthChromosome required. LengthChromosome='+str(LCmin)+' is taken.')
    else:print('LengthChromosome='+str(LengthChromosome)+' is taken.')
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    Pob_2=np.zeros((Nr*ne,LC),dtype=int).tolist()
    for i in range(Nr*ne):
        for j in range(LC):
                Pob_2[i][j]=np.random.randint(2)
    Pob_10= fPobDec(Pob_2,ne,Nr)
    Pob_float = fScale(Pob_10,lim,ne,LC,Nr,ndecimals=ndecimals4float)
    Pob_2=[]
    Pob_10=[]
    SpecialCrossoverOperator='SBX'
    print("SpecialCrossoverOperator='SBX'")
    
    #Initialize parameters
    Nr00=Nr
    Nrc=0
    Pob_floatc=[]
    fxc=[]
    fxpc=[]
    punishc=[]
    Pob_2c=[]
    Nraux=Nr
    lenfront0=1
    change=0
    front=[[]]
    lenindc2=0
    Nrelite=NumIndiv
    stop=0
    tstart = time.time()
    #Main loop
    contG=0
    while contG<NGenerations:
        if contG>1 and not isinstance(Nstopelite,bool) and stop>=Nstopelite:
            print('\rstop')
            break
        fx,breaking=ObjSpace(Pob_float)
        if breaking==True:
            print('fx error',fx)
            break
        contG+=1
        
        
        fxt=fx+fxc
        Nrt=Nr+Nrc
        Pob_floatt=Pob_float+Pob_floatc
        
        #Algorithm1
        indcv=np.ones((1,Nrt), dtype=bool)[0]
        if Nrc>=Nrt/4:
            indcv,lim=Distance_parameter_filter(Pob_floatt,ne,contG,lim,Pob_floatc,Nrt,Nrc,indcv,limCV=limCV*np.random.rand(),epsilon=epsilon_bar)
            if len(indcv)<4:indcv=np.arange(Nrt)
        else:
            indcv=np.arange(Nrt)#[indcv]
        #Algorithm2
        front,Pcontreal,Pob_floatt,fxt=approximate_non_dominated_sorting(indcv,lim,ne,fxt,multiobjectives,epsilon_bar,Nrelite,Pob_floatt)
        
        #Algorithm6
        if (len(front[0])>Nr/4) or len(front[0])>Nrt*0.8:
            front0,front1,Pcontreal=Outlier_filter(front[0],front[1],Nrelite,Nrt,multiobjectives,np.array(fxt)[front[0]])
            front[0]=front0
            front[1]=front1
        if Pcontreal<Nrelite:stop2=0
        indsort=(np.array(front[0])[:min(Nrt,len(front[0]))]).tolist()\
                    +(np.array(front[1])[:Nrt-len(front[0])]).tolist()
        indc=front[0]
        
        
        
        
        Nrc=min(Nrelite+Nr00,len(indc))
        fxc,Pob_floatc=Objects(Nrt,Nrc,ne,multiobjectives,indc,fxt,Pob_floatt)
        Nr=max(10,min(Nrt-Nrt%2,Nr00,Nr00-(Nrc-Nrelite)-(Nr00-(Nrc-Nrelite))%2))
        indivconsidered=np.zeros((1,Nrt),dtype=bool)[0]
        indivconsidered[indsort]=True
        decr=False
        
        
        ind=np.where(indivconsidered==1)[0]
        Nr=min(Nr,len(ind))
        Pob_float=np.zeros((1,Nr*ne))[0].tolist()
        for fil in range(Nr):
            Pob_float[fil*ne:(fil+1)*ne]=Pob_floatt[ind[fil]*ne:(ind[fil]+1)*ne]
            
        whererank1=front[0]
        minfx=[[0 for ill in range(multiobjectives)] for il in range(len(whererank1))]
        Pob_floatmax=[[0 for il in range(ne)] for ill in range(len(whererank1))]
        for fil in range(len(whererank1)):
            indmax=whererank1[fil]
            ri=(indmax)*ne;rj=(indmax+1)*ne
            Pob_floatmax[fil][:]=copy.deepcopy(Pob_floatt[ri:rj])
            minfx[fil][:]=copy.deepcopy(fxt[indmax])
        lenunique=Pcontreal
        
        
        if contG==1 or contG%printcont==0:
            TEXT='\r'
            TEXT+='len(front[0])='+str(len(front[0]))
            TEXT+='unique='+str(lenunique)
            TEXT+=' EndGeneration='+str(contG)+'                     '
            sys.stdout.write(TEXT)
            sys.stdout.flush()
        
        if binary:
            Pob_10=fScalInv(Pob_float,lim,ne,LC,Nr);Pob_2=fPobDecInv(Pob_10,ne,Nr,LC)
            Pob_2,Parents,Couples=freproduction( Pob_2,Nr,ne,decr,LC,SpecialCrossoverOperator=SpecialCrossoverOperator)
            Pob_2=fMutation(Pob_2,LC,MutationPorc)
            Pob_10= fPobDec(Pob_2,ne,Nr)
            Pob_float = fScale(Pob_10,lim,ne,LC,Nr,ndecimals=ndecimals4float)
        else:
            Pob_float,Parents,Couples=Crossover_mutationELY2( Pob_float,Nr,ne,LC,MutationPorc,lim)
        if lenunique<=lenfront0:
            stop+=1
        else:
            stop=0
        lenfront0=max(lenfront0,lenunique)
        
    runtime = time.time() - tstart
    print('\n elapsed [s]=',runtime)
    print('Pob_floatmax[0]=',Pob_floatmax[0])
    return Pob_floatmax,minfx,runtime

# %% NSGAII

def CrowdingDistance(front,rank,multiobjectives,Nr,fx):
    index_of_fronts=np.argsort(rank)
    sorted_based_on_front=np.array(fx)[index_of_fronts,:]
    inv_index_of_fronts=np.array(range(len(rank)))[index_of_fronts]
    current_index = 0
    crowdingdistance=np.zeros((Nr,1))
    for F in range(len(front)-1):
        distance=0
        y=np.zeros((len(front[F]),multiobjectives))
        previous_index=current_index
        for i in range(len(front[F])):
            y[i,:]=sorted_based_on_front[current_index+i,:]
        current_index+=i+1
        for i in range(multiobjectives):
            index_of_objectives=np.argsort(y[:,i])
            sorted_based_on_objective=y[index_of_objectives,:]
            f_max=sorted_based_on_objective[-1,i]
            f_min=sorted_based_on_objective[0,i]
            y[index_of_objectives[-1],i]=np.Inf#1#
            y[index_of_objectives[0],i]=np.Inf#1#
            for j in range(1,len(index_of_objectives)-1):
                next_obj=sorted_based_on_objective[j+1,i]
                previous_obj=sorted_based_on_objective[j-1,i]
                if f_max-f_min==0:
                    y[index_of_objectives[j],i]=np.Inf#1#
                else:
                    y[index_of_objectives[j],i]=(next_obj-previous_obj)/(f_max-f_min)
        distance=np.zeros((len(front[F]),1))
        for i in range(multiobjectives):
            distance[:,0]+=y[:,i]
        crowdingdistance[previous_index:current_index,:]=distance#y
    crowdingdistance=crowdingdistance[inv_index_of_fronts]
    return crowdingdistance
def ffastnondominatedsort(Nr,multiobjectives,fx,ne,Pob_float):
    
    individualsthatdominatethis=[0 for il in range(Nr)]
    individualsdominated=[[] for il in range(Nr)]
    rank=[0 for _ in range(Nr)]
    F=1
    front=[[]]
    for i in range(Nr):
        for j in range(Nr):
            if i!=j:
                this_dom=0
                dom_equal=0
                dom_to_this=0
                for k in range(multiobjectives):
                    if fx[i][k]<fx[j][k]:
                        this_dom+=1
                    elif fx[i][k]==fx[j][k]:
                        dom_equal+=1
                    else:
                        dom_to_this+=1
                if this_dom==0 and dom_equal!=multiobjectives:
                    individualsthatdominatethis[i]+=1
                elif dom_to_this==0 and dom_equal!=multiobjectives:
                    individualsdominated[i].append(j)
        if individualsthatdominatethis[i]==0:
            rank[i]=1
            front[F-1].append(i)
    while len(front[F-1])>0:
        Q=[]
        for i in range(len(front[F-1])):
            if len(individualsdominated[front[F-1][i]])>0:
                for j in range(len(individualsdominated[front[F-1][i]])):
                    individualsthatdominatethis[individualsdominated[front[F-1][i]][j]]-=1
                    if individualsthatdominatethis[individualsdominated[front[F-1][i]][j]]==0:
                        rank[individualsdominated[front[F-1][i]][j]]=F+1
                        Q.append(individualsdominated[front[F-1][i]][j])
        F+=1
        front.append(Q)
    front=front[:len(front)-1]
    return rank,front
def fSelectionTourNSGAII(Nr,nt,rank,crowdingdistance):
    candidate=np.zeros((nt,1))-1
    c_obj_rank=np.zeros((nt,1))
    c_obj_distance=np.zeros((nt,1))
    decr=[0 for _ in range(Nr)]
    
    if Nr==len(rank):
        indivconsidered=np.array([1 for _ in range(Nr)])
    else:
        indivconsidered=np.array([0 for _ in range(len(rank))])
        indivconsidered[np.random.permutation(np.arange(len(rank)))[:Nr]]=1
        rank=np.array(rank)[indivconsidered]
        crowdingdistance=crowdingdistance[indivconsidered]
    while sum(decr)<Nr:
        i=0
        a=int(np.floor((Nr)*np.random.rand()))
        candidate[i]=a
        c_obj_rank[i,:]=rank[a]
        c_obj_distance[i,:]=crowdingdistance[a]
        while i<nt-1:
            a=int(np.floor((Nr)*np.random.rand()))
            if (candidate!=a).all():
                i+=1
                candidate[i]=a
                c_obj_rank[i,:]=rank[a]
                c_obj_distance[i,:]=crowdingdistance[a]
        max_candidate=0
        min_candidate=np.where(c_obj_rank==np.min(c_obj_rank))[0]
        if len(min_candidate)>1:
            max_candidate=np.where(c_obj_distance[min_candidate]==np.max(c_obj_distance[min_candidate]))[0]
            if len(max_candidate)>1:
                max_candidate=max_candidate[0]
        j=min_candidate[max_candidate]
        decr[int(candidate[j])]+=1
        candidate=np.zeros((nt,1))-1
    return decr,indivconsidered,rank,crowdingdistance
def Crossover_mutationNoBinary(Pob_float,decr,MutationPorc,ne,Nr,lim,mu=0.2,onlycrossover=False,onlymutation=False):
    Pob_C=[0 for _ in range(len(Pob_float))]
    contindi=-1
    mum=MutationPorc/100
    if Nr==1:
        indparent_3=0
        child_3=Pob_float[indparent_3*ne:(indparent_3+1)*ne]
        for D in range(ne):
            rj=np.random.rand()
            if rj<0.5:
                delta=(2*rj)**(1/(mum+1)) - 1
            else:
                delta= 1 - (2*(1 - rj))**(1/(mum+1))
            child_3[D]=min(lim[D][1],max(lim[D][0],child_3[D]+delta))
        contindi+=1
        Pob_C[contindi*ne:(contindi+1)*ne]=child_3
    else:
        for indiv in range(Nr):
            inddcr=np.arange(Nr)[np.array(decr)>0]
            if len(inddcr)==0:
                break
            if onlycrossover==True or (onlymutation==False and np.random.rand()<=0.9 and len(inddcr)>=2):
                indparent_1,indparent_2=np.random.choice(inddcr,2,replace=False)
                decr[indparent_1]-=1
                # indparent_2=np.random.choice(inddcr)
                decr[indparent_2]-=1
                parent_1=Pob_float[indparent_1*ne:(indparent_1+1)*ne]
                parent_2=Pob_float[indparent_2*ne:(indparent_2+1)*ne]
                child_1=[0 for _ in range(ne)]
                child_2=[0 for _ in range(ne)]
                for D in range(ne):
                    u=np.random.rand()
                    if u<=0.5:
                        bq=(2*u)**(1/(mu+1))
                    else:
                        bq=(1/(2*(1-u)))**(1/(mu+1))
                    child_1[D]=min(lim[D][1],max(lim[D][0],0.5*(((1 + bq)*parent_1[D]) + (1 - bq)*parent_2[D])))
                    child_2[D]=min(lim[D][1],max(lim[D][0],0.5*(((1 - bq)*parent_1[D]) + (1 + bq)*parent_2[D])))
                was_crossover = 1
                was_mutation = 0
            else:
                indparent_3=np.random.choice(inddcr)
                child_3=Pob_float[indparent_3*ne:(indparent_3+1)*ne]
                decr[indparent_3]-=1
                for D in range(ne):
                    rj=np.random.rand()
                    if rj<0.5:
                        delta=(2*rj)**(1/(mum+1)) - 1
                    else:
                        delta= 1 - (2*(1 - rj))**(1/(mum+1))
                    child_3[D]=min(lim[D][1],max(lim[D][0],child_3[D]+delta))
                was_mutation = 1;
                was_crossover = 0;
            if was_crossover:
                contindi+=1
                Pob_C[contindi*ne:(contindi+1)*ne]=child_1
                contindi+=1
                Pob_C[contindi*ne:(contindi+1)*ne]=child_2
            elif was_mutation:
                contindi+=1
                Pob_C[contindi*ne:(contindi+1)*ne]=child_3
    Pob_float=Pob_C.copy()
    return Pob_float
def ffastnondominatedsortELY(Nr,multiobjectives,fx,ne,Pob_float,Nrconsidered):

    rank=[1 for _ in range(Nr)]
    for i in range(Nr-1):
        for j in range(i+1,Nr):
            this_dom=0
            dom_to_this=0
            for k in range(multiobjectives):
                if fx[i][k]<fx[j][k]:
                    this_dom=1
                elif fx[i][k]>fx[j][k]:
                    dom_to_this=1
            if this_dom-dom_to_this>0:
                rank[j]+=1
            elif dom_to_this-this_dom>0:
                rank[i]+=1

    front=[]
    lenT=0
    i=0
    while lenT<min(Nr,Nrconsidered):
        i+=1
        if rank.__contains__(i):
            ind=np.where(np.array(rank)==(i))[0]
            front.append(ind.tolist())
            lenT+=len(ind)

    if lenT<Nr:
        front.append(np.where(np.array(rank)>i)[0].tolist())

    return rank,front

def NSGAII(functionanalysis,lim,NumIndiv,MutationPorc,NGenerations,Nstopelite=10,multiobjectives=3,\
        LengthChromosome=0,ndecimals4float=10,printcont=1,epsilon_bar=False,binary=False):
    #Objective function
    functionanalysis.index('.');analysismodule='';aux=0;function=''
    for i in range(len(str(functionanalysis))):
        if functionanalysis[i]=='.':aux=1
        if aux==0:
            analysismodule=analysismodule+functionanalysis[i]
        elif functionanalysis[i]!='.':
           function=function+functionanalysis[i]
    if function!='py':
        Module = importlib.import_module(analysismodule)
        method=getattr(Module,function)
        
    def ObjSpace(Pob_float):
        results=[]
        Nr11=int(len(Pob_float)/ne)
        resultsi=[]
        breaking=False
        for i in range(Nr11):
            ri=(i)*ne
            rj=(i+1)*ne
            x=Pob_float[ri:rj]
            resultsi=method(x)
            if resultsi[0]==0:
                breaking=True
            results.append(resultsi)
        fx=[results[il][0] for il in range(len(results))]
        return fx,breaking
        
    #Initial random population
    Nr=NumIndiv
    lim=[copy.deepcopy(lim[i]) for i in range(len(lim))]
    
    intfloat=[lim[i][2] for i in range(len(lim))]
    if LengthChromosome==0 and intfloat.__contains__('float'):LengthChromosome=40
    LCmin=max(5,len(bin(int(round(max([lim[i][1]-lim[i][0] for i in range(len(lim))]),0)+1)))-1)
    if LengthChromosome==[] or LengthChromosome=='':LengthChromosome=0
    if LCmin>LengthChromosome:print('LengthChromosome less than LengthChromosome required. LengthChromosome='+str(LCmin)+' is taken.')
    else:print('LengthChromosome='+str(LengthChromosome)+' is taken.')
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    Pob_2=np.zeros((Nr*ne,LC),dtype=int).tolist()
    for i in range(Nr*ne):
        for j in range(LC):
                Pob_2[i][j]=np.random.randint(2)
    Pob_10= fPobDec(Pob_2,ne,Nr)
    Pob_float = fScale(Pob_10,lim,ne,LC,Nr,ndecimals=ndecimals4float)
    Pob_2=[]
    Pob_10=[]
    SpecialCrossoverOperator='SBX'
    print("SpecialCrossoverOperator='SBX'")
    
    #Initialize parameters
    Nr00=Nr
    Nrc=0
    Pob_floatc=[]
    fxc=[]
    fxpc=[]
    punishc=[]
    Pob_2c=[]
    Nraux=Nr
    lenfront0=1
    change=0
    front=[[]]
    lenindc2=0
    Nrelite=NumIndiv
    stop=0
    indivcompiting=2
    tstart = time.time()
    #Main loop
    contG=0
    while contG<NGenerations:
        if contG>1 and not isinstance(Nstopelite,bool) and stop>=Nstopelite:
            print('\rstop')
            break
        fx,breaking=ObjSpace(Pob_float)
        if breaking==True:
            print('fx error',fx)
            break
        contG+=1
        
        
        fxt=fx+fxc
        Nrt=Nr+Nrc
        Pob_floatt=Pob_float+Pob_floatc
        # rank,front=ffastnondominatedsort(Nrt,multiobjectives,fxt,ne,Pob_floatt)
        rank,front=ffastnondominatedsortELY(Nrt,multiobjectives,fxt,ne,Pob_floatt,Nrt-1)
        crowdingdistance=CrowdingDistance(front,rank,multiobjectives,Nrt,fxt)
        front=[(np.array(front[il])[np.argsort((crowdingdistance[front[il]].T)[0])[::-1]]).tolist() for il in range(len(front))]
        indc=[front[il][ill] for il in range(len(front)) for ill in range(len(front[il]))]        
        
        Nrc=min(Nrelite+Nr00,len(indc))
        fxc,Pob_floatc=Objects(Nrt,Nrc,ne,multiobjectives,indc,fxt,Pob_floatt)
        
        indc=indc[:Nrc]
        rank=np.array(rank)[indc];crowdingdistance=crowdingdistance[indc]
        indivconsidered=np.array([0 for _ in range(Nrt)])
        indivconsidered[indc]=1
        decr,_,rank,crowdingdistance=fSelectionTourNSGAII(Nr,indivcompiting,rank,crowdingdistance)
        
        
        ind=np.where(indivconsidered==1)[0]
        Pob_float=np.zeros((1,Nr*ne))[0].tolist()
        for fil in range(min(Nr,len(ind))):
            Pob_float[fil*ne:(fil+1)*ne]=Pob_floatt[ind[fil]*ne:(ind[fil]+1)*ne]
            
        whererank1=front[0]
        minfx=[[0 for ill in range(multiobjectives)] for il in range(len(whererank1))]
        Pob_floatmax=[[0 for il in range(ne)] for ill in range(len(whererank1))]
        for fil in range(len(whererank1)):
            indmax=whererank1[fil]
            ri=(indmax)*ne;rj=(indmax+1)*ne
            Pob_floatmax[fil][:]=copy.deepcopy(Pob_floatt[ri:rj])
            minfx[fil][:]=copy.deepcopy(fxt[indmax])
        lenunique=len(front[0])
        
        
        if contG==1 or contG%printcont==0:
            TEXT='\r'
            TEXT+='len(front[0])='+str(len(front[0]))
            TEXT+='unique='+str(lenunique)
            TEXT+=' EndGeneration='+str(contG)+'                     '
            sys.stdout.write(TEXT)
            sys.stdout.flush()
        
        if binary:
            Pob_10=fScalInv(Pob_float,lim,ne,LC,Nr);Pob_2=fPobDecInv(Pob_10,ne,Nr,LC)
            Pob_2,Parents,Couples=freproduction( Pob_2,Nr,ne,decr,LC,SpecialCrossoverOperator=SpecialCrossoverOperator)
            Pob_2=fMutation(Pob_2,LC,MutationPorc)
            Pob_10= fPobDec(Pob_2,ne,Nr)
            Pob_float = fScale(Pob_10,lim,ne,LC,Nr,ndecimals=ndecimals4float)
        else:
            Pob_float=Crossover_mutationNoBinary(Pob_float,decr,MutationPorc,ne,Nr,lim)
        if lenunique<=lenfront0:
            stop+=1
        else:
            stop=0
        lenfront0=max(lenfront0,lenunique)
        
    runtime = time.time() - tstart
    print('\n elapsed [s]=',runtime)
    print('Pob_floatmax[0]=',Pob_floatmax[0])
    return Pob_floatmax,minfx,runtime
#%% SDNSGAII
def Crossover_mutationNoBinaryBLX(Pob_float,decr,MutationPorc,ne,Nr,lim,mu=0.2,onlycrossover=False,onlymutation=False,alpha=0.5):
    Pob_C=[0 for _ in range(len(Pob_float))]
    contindi=-1
    mum=MutationPorc/100
    if Nr==1:
        indparent_3=0
        child_3=Pob_float[indparent_3*ne:(indparent_3+1)*ne]
        for D in range(ne):
            rj=np.random.rand()
            if rj<0.5:
                delta=(2*rj)**(1/(mum+1)) - 1
            else:
                delta= 1 - (2*(1 - rj))**(1/(mum+1))
            child_3[D]=min(lim[D][1],max(lim[D][0],child_3[D]+delta))
        contindi+=1
        Pob_C[contindi*ne:(contindi+1)*ne]=child_3
    else:
        for indiv in range(Nr):
            inddcr=np.arange(Nr)[np.array(decr)>0]
            if len(inddcr)==0:
                break
            if onlycrossover==True or (onlymutation==False and np.random.rand()<=0.9 and len(inddcr)>=2):
                indparent_1,indparent_2=np.random.choice(inddcr,2,replace=False)
                decr[indparent_1]-=1
                # indparent_2=np.random.choice(inddcr)
                decr[indparent_2]-=1
                parent_1=Pob_float[indparent_1*ne:(indparent_1+1)*ne]
                parent_2=Pob_float[indparent_2*ne:(indparent_2+1)*ne]
                child_1=[0 for _ in range(ne)]
                child_2=[0 for _ in range(ne)]
                for D in range(ne):
                    distance=abs(parent_2[D]-parent_1[D])
                    l=min(parent_2[D],parent_1[D])-alpha*distance
                    u=max(parent_2[D],parent_1[D])+alpha*distance
                    child_1[D]=min(lim[D][1],max(lim[D][0], l+np.random.rand()*(u-l) ))
                    child_2[D]=min(lim[D][1],max(lim[D][0], l+np.random.rand()*(u-l) ))
                was_crossover = 1
                was_mutation = 0
            else:
                indparent_3=np.random.choice(inddcr)
                child_3=Pob_float[indparent_3*ne:(indparent_3+1)*ne]
                decr[indparent_3]-=1
                for D in range(ne):
                    distance=abs(child_3[D])*mum
                    l=child_3[D]-alpha*distance
                    u=child_3[D]+alpha*distance
                    child_3[D]=min(lim[D][1],max(lim[D][0], l+np.random.rand()*(u-l) ))
                    # child_3[D]=min(lim[D][1],max(lim[D][0],child_3[D]+delta))
                was_mutation = 1;
                was_crossover = 0;
            if was_crossover:
                contindi+=1
                Pob_C[contindi*ne:(contindi+1)*ne]=child_1
                contindi+=1
                Pob_C[contindi*ne:(contindi+1)*ne]=child_2
            elif was_mutation:
                contindi+=1
                Pob_C[contindi*ne:(contindi+1)*ne]=child_3
    Pob_float=Pob_C.copy()
    return Pob_float
def CrowdingDistanceDesicionSpace(front,rank,multiobjectives,Nr,Pob_float):
    ne=int(len(Pob_float)/Nr)
    Pob_floatMatrix=np.array([copy.deepcopy(Pob_float[il*ne:(il+1)*ne]) for il in range(Nr)])
    
    index_of_fronts=np.argsort(rank)
    sorted_based_on_front=np.array(Pob_floatMatrix)[index_of_fronts,:]
    inv_index_of_fronts=np.array(range(len(rank)))[index_of_fronts]
    current_index = 0
    crowdingdistanceDesicionSpace=np.zeros((Nr,1))
    for F in range(len(front)-1):
        distance=0
        y=np.zeros((len(front[F]),multiobjectives))
        previous_index=current_index
        for i in range(len(front[F])):
            try:
                y[i,:]=sorted_based_on_front[current_index+i,:]
            except:
                e=100
        current_index+=i+1
        for i in range(multiobjectives):
                index_of_objectives=np.argsort(y[:,i])
        
                sorted_based_on_objective=y[index_of_objectives,:]
                f_max=sorted_based_on_objective[-1,i]
                f_min=sorted_based_on_objective[0,i]
                y[index_of_objectives[-1],i]=np.Inf
                y[index_of_objectives[0],i]=np.Inf
                for j in range(1,len(index_of_objectives)-1):
                    next_obj=sorted_based_on_objective[j+1,i]
                    previous_obj=sorted_based_on_objective[j-1,i]
                    if f_max-f_min==0:
                        y[index_of_objectives[j],i]=np.Inf
                    else:
                        y[index_of_objectives[j],i]=(next_obj-previous_obj)/(f_max-f_min)
        distance=np.zeros((len(front[F]),1))
        for i in range(multiobjectives):
            distance[:,0]+=y[:,i]
        crowdingdistanceDesicionSpace[previous_index:current_index,:]=distance
    crowdingdistanceDesicionSpace=crowdingdistanceDesicionSpace[inv_index_of_fronts]
    return crowdingdistanceDesicionSpace
def fSelectionTourASDNSGAII(Nr,rank,crowdingdistance,crowdingdistanceDesicionSpace,nt=2):
    candidate=np.zeros((nt,1))-1
    c_obj_rank=np.zeros((nt,1))
    c_obj_distance=np.zeros((nt,1))
    decr=[0 for _ in range(Nr)]
    indivconsidered=[0 for _ in range(Nr)]
    if Nr==len(rank):
        indivconsidered=np.array([1 for _ in range(Nr)])
    else:
        indivconsidered=np.array([0 for _ in range(len(rank))])
        indivconsidered[np.random.permutation(np.arange(len(rank)))[:Nr]]=1
        rank=np.copy(rank)[indivconsidered]
        crowdingdistance=np.copy(crowdingdistance)[indivconsidered]
        crowdingdistanceDesicionSpace=np.copy(crowdingdistanceDesicionSpace)[indivconsidered]
    CDavgx=np.average(crowdingdistanceDesicionSpace)
    CDavgf=np.average(crowdingdistance)
    while sum(decr)<Nr:
        i=0
        a=int(np.floor((Nr)*np.random.rand()))
        candidate[i]=a
        c_obj_rank[i,:]=rank[a]
        CDx=crowdingdistanceDesicionSpace[a]
        CDf=crowdingdistance[a]
        if CDx>CDavgx or CDf>CDavgf:SCD=max(CDx,CDf)
        else:SCD=min(CDx,CDf)
        c_obj_distance[i,:]=SCD
        while i<nt-1:
            a=int(np.floor((Nr)*np.random.rand()))
            if (candidate!=a).all():
                i+=1
                candidate[i]=a
                c_obj_rank[i,:]=rank[a]
                if CDx>CDavgx or CDf>CDavgf:SCD=max(CDx,CDf)
                else:SCD=min(CDx,CDf)
                c_obj_distance[i,:]=SCD
        max_candidate=0
        min_candidate=np.where(c_obj_rank==np.min(c_obj_rank))[0]
        if len(min_candidate)>1:
            max_candidate=np.where(c_obj_distance[min_candidate]==np.max(c_obj_distance[min_candidate]))[0]
            if len(max_candidate)>1:
                max_candidate=max_candidate[0]
        j=min_candidate[max_candidate]
        decr[int(candidate[j])]+=1
        candidate=(np.zeros((1,nt),dtype=int)[0])-1
    return decr,indivconsidered,rank,crowdingdistance,crowdingdistanceDesicionSpace
def freproduction(Pob_2,Nr,ne,decr,LC,SpecialCrossoverOperator=False):#cross-over function
    import copy    
    import numpy as np
    if decr!=False:
        dec=copy.deepcopy(decr)
        Parents=[0 for _ in range(Nr)]
        cont=0
        while cont<Nr:
            v=max(dec)
            ind=dec.index(v)
            Parents[cont:cont+v]=[ind]*v
            dec[ind]=0
            cont+=v
        t1=np.random.permutation(Nr)
        if len(t1)%2!=0:print('The Number of Individuals must be even')
        t2=[[t1[i],t1[i+1]] for i in range(0,len(t1),2)]
        Couples=np.zeros((len(t2),2)).tolist()
        for i in range(len(t2)):
            Couples[i][0]=Parents[t2[i][0]]
            Couples[i][1]=Parents[t2[i][1]]
        
        Pob_C=[0 for _ in range(len(Pob_2))]
        a1=[0 for _ in range(len(t2))]
        a2=[0 for _ in range(len(t2))]
        for t in range(len(t2)):
            a1[t]=Couples[t][0]
            a2[t]=Couples[t][1]
            for i in range(ne):
                if SpecialCrossoverOperator=='SBX':
                    pcross=int(SBX(nc=2)*LC)
                else:
                    pcross=np.random.randint(0,LC)
                r=(a1[t])*ne+i
                aux1=Pob_2[r][0:pcross]
                aux2=Pob_2[r][pcross:LC]
                r1=(a2[t])*ne+i
                aux3=Pob_2[r1][0:pcross]
                aux4=Pob_2[r1][pcross:LC]
                
                Pob_C[t*ne+i]=aux1+aux4
                Pob_C[int(len(Pob_2)/2)+t*ne+i]=aux3+aux2
        return Pob_C,Parents,Couples
    else:
        return Pob_2,0,0
def fMutation(Pob_2,LC,MutationPorc):
    import numpy as np
    R=len(Pob_2)
    C=len(Pob_2[0])
    Tgenes=R*C
    GenesMut=round(Tgenes*MutationPorc/100)
    for i in range(GenesMut):
        row=np.random.randint(0,R)
        column=np.random.randint(0,C)
        Pob_2[row][column]=1-Pob_2[row][column]
    return Pob_2
def SDNSGAII(functionanalysis,lim,NumIndiv,MutationPorc,NGenerations,Nstopelite=10,multiobjectives=3,\
        LengthChromosome=0,ndecimals4float=10,printcont=1,epsilon_bar=False,binary=False):
    #Objective function
    functionanalysis.index('.');analysismodule='';aux=0;function=''
    for i in range(len(str(functionanalysis))):
        if functionanalysis[i]=='.':aux=1
        if aux==0:
            analysismodule=analysismodule+functionanalysis[i]
        elif functionanalysis[i]!='.':
           function=function+functionanalysis[i]
    if function!='py':
        Module = importlib.import_module(analysismodule)
        method=getattr(Module,function)
        
    def ObjSpace(Pob_float):
        results=[]
        Nr11=int(len(Pob_float)/ne)
        resultsi=[]
        breaking=False
        for i in range(Nr11):
            ri=(i)*ne
            rj=(i+1)*ne
            x=Pob_float[ri:rj]
            resultsi=method(x)
            if resultsi[0]==0:
                breaking=True
            results.append(resultsi)
        fx=[results[il][0] for il in range(len(results))]
        return fx,breaking
        
    #Initial random population
    Nr=NumIndiv
    lim=[copy.deepcopy(lim[i]) for i in range(len(lim))]
    
    intfloat=[lim[i][2] for i in range(len(lim))]
    if LengthChromosome==0 and intfloat.__contains__('float'):LengthChromosome=40
    LCmin=max(5,len(bin(int(round(max([lim[i][1]-lim[i][0] for i in range(len(lim))]),0)+1)))-1)
    if LengthChromosome==[] or LengthChromosome=='':LengthChromosome=0
    if LCmin>LengthChromosome:print('LengthChromosome less than LengthChromosome required. LengthChromosome='+str(LCmin)+' is taken.')
    else:print('LengthChromosome='+str(LengthChromosome)+' is taken.')
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    Pob_2=np.zeros((Nr*ne,LC),dtype=int).tolist()
    for i in range(Nr*ne):
        for j in range(LC):
                Pob_2[i][j]=np.random.randint(2)
    Pob_10= fPobDec(Pob_2,ne,Nr)
    Pob_float = fScale(Pob_10,lim,ne,LC,Nr,ndecimals=ndecimals4float)
    Pob_2=[]
    Pob_10=[]
    SpecialCrossoverOperator='SBX'
    print("SpecialCrossoverOperator='SBX'")
    
    #Initialize parameters
    Nr00=Nr
    Nrc=0
    Pob_floatc=[]
    fxc=[]
    fxpc=[]
    punishc=[]
    Pob_2c=[]
    Nraux=Nr
    lenfront0=1
    change=0
    front=[[]]
    lenindc2=0
    Nrelite=NumIndiv
    stop=0
    indivcompiting=2
    tstart = time.time()
    #Main loop
    contG=0
    while contG<NGenerations:
        if contG>1 and not isinstance(Nstopelite,bool) and stop>=Nstopelite:
            print('\rstop')
            break
        fx,breaking=ObjSpace(Pob_float)
        if breaking==True:
            print('fx error',fx)
            break
        contG+=1
        
        
        fxt=fx+fxc
        Nrt=Nr+Nrc
        Pob_floatt=Pob_float+Pob_floatc
        rank,front=ffastnondominatedsortELY(Nrt,multiobjectives,fxt,ne,Pob_floatt,Nrt-1)
        crowdingdistance=CrowdingDistance(front,rank,multiobjectives,Nrt,fxt)
        crowdingdistanceDesicionSpace=CrowdingDistanceDesicionSpace(front,rank,ne,Nrt,Pob_floatt)
        front=[(np.array(front[il])[np.argsort((crowdingdistance[front[il]].T)[0])[::-1]]).tolist() for il in range(len(front))]
        indc=[front[il][ill] for il in range(len(front)) for ill in range(len(front[il]))]        
        
        Nrc=min(Nrelite+Nr00,len(indc))
        fxc,Pob_floatc=Objects(Nrt,Nrc,ne,multiobjectives,indc,fxt,Pob_floatt)
        
        indc=indc[:Nrc]
        rank=np.array(rank)[indc];crowdingdistance=crowdingdistance[indc]
        indivconsidered=np.array([0 for _ in range(Nrt)])
        indivconsidered[indc]=1
        decr,_,rank,crowdingdistance,crowdingdistanceDesicionSpace=\
            fSelectionTourASDNSGAII(Nr,rank,crowdingdistance,crowdingdistanceDesicionSpace,nt=indivcompiting)
        
        
        ind=np.where(indivconsidered==1)[0]
        Pob_float=np.zeros((1,Nr*ne))[0].tolist()
        for fil in range(min(Nr,len(ind))):
            Pob_float[fil*ne:(fil+1)*ne]=Pob_floatt[ind[fil]*ne:(ind[fil]+1)*ne]
            
        whererank1=front[0]
        minfx=[[0 for ill in range(multiobjectives)] for il in range(len(whererank1))]
        Pob_floatmax=[[0 for il in range(ne)] for ill in range(len(whererank1))]
        for fil in range(len(whererank1)):
            indmax=whererank1[fil]
            ri=(indmax)*ne;rj=(indmax+1)*ne
            Pob_floatmax[fil][:]=copy.deepcopy(Pob_floatt[ri:rj])
            minfx[fil][:]=copy.deepcopy(fxt[indmax])
        lenunique=len(front[0])
        
        
        if contG==1 or contG%printcont==0:
            TEXT='\r'
            TEXT+='len(front[0])='+str(len(front[0]))
            TEXT+='unique='+str(lenunique)
            TEXT+=' EndGeneration='+str(contG)+'                     '
            sys.stdout.write(TEXT)
            sys.stdout.flush()
        
        if binary:
            Pob_10=fScalInv(Pob_float,lim,ne,LC,Nr);Pob_2=fPobDecInv(Pob_10,ne,Nr,LC)
            Pob_2,Parents,Couples=freproduction( Pob_2,Nr,ne,decr,LC,SpecialCrossoverOperator=SpecialCrossoverOperator)
            Pob_2=fMutation(Pob_2,LC,MutationPorc)
            Pob_10= fPobDec(Pob_2,ne,Nr)
            Pob_float = fScale(Pob_10,lim,ne,LC,Nr,ndecimals=ndecimals4float)
        else:
            Pob_float=Crossover_mutationNoBinaryBLX(Pob_float,decr,MutationPorc,ne,Nr,lim,alpha=0.5)
        
        if lenunique<=lenfront0:
            stop+=1
        else:
            stop=0
        lenfront0=max(lenfront0,lenunique)
        
    runtime = time.time() - tstart
    print('\n elapsed [s]=',runtime)
    print('Pob_floatmax[0]=',Pob_floatmax[0])
    return Pob_floatmax,minfx,runtime
#%% I-MOEA/D
def CreateSubProblems(multiobjectives,Nr,T):
    sp_lambda=np.zeros((Nr,multiobjectives))
    sp_neighbors=np.zeros((Nr,T),dtype=int)
    for fil in range(Nr):
        lambda_=np.random.rand(multiobjectives)
        sp_lambda[fil]=lambda_#/np.linalg.norm(lambda_)
    D=sp.spatial.distance.cdist(sp_lambda, sp_lambda)
    for fil in range(Nr):
        SO=np.argsort(D[fil,:])
        sp_neighbors[fil,:]=SO[:T]
    return sp_lambda,sp_neighbors
def IMOEAD(functionanalysis,lim,NumIndiv,MutationPorc,NGenerations,Nstopelite=10,multiobjectives=3,\
        LengthChromosome=0,ndecimals4float=10,printcont=1,epsilon_bar=False,considerfront=False,binary=False):
    #Objective function
    functionanalysis.index('.');analysismodule='';aux=0;function=''
    for i in range(len(str(functionanalysis))):
        if functionanalysis[i]=='.':aux=1
        if aux==0:
            analysismodule=analysismodule+functionanalysis[i]
        elif functionanalysis[i]!='.':
           function=function+functionanalysis[i]
    if function!='py':
        Module = importlib.import_module(analysismodule)
        method=getattr(Module,function)
        
    def ObjSpace(Pob_float):
        results=[]
        Nr11=int(len(Pob_float)/ne)
        resultsi=[]
        breaking=False
        for i in range(Nr11):
            ri=(i)*ne
            rj=(i+1)*ne
            x=Pob_float[ri:rj]
            resultsi=method(x)
            if resultsi[0]==0:
                breaking=True
            results.append(resultsi)
        fx=[results[il][0] for il in range(len(results))]
        return fx,breaking
        
    #Initial random population
    Nr=NumIndiv
    lim=[copy.deepcopy(lim[i]) for i in range(len(lim))]
    
    intfloat=[lim[i][2] for i in range(len(lim))]
    if LengthChromosome==0 and intfloat.__contains__('float'):LengthChromosome=40
    LCmin=max(5,len(bin(int(round(max([lim[i][1]-lim[i][0] for i in range(len(lim))]),0)+1)))-1)
    if LengthChromosome==[] or LengthChromosome=='':LengthChromosome=0
    if LCmin>LengthChromosome:print('LengthChromosome less than LengthChromosome required. LengthChromosome='+str(LCmin)+' is taken.')
    else:print('LengthChromosome='+str(LengthChromosome)+' is taken.')
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    LC=max(LengthChromosome,LCmin)
    ne=len(lim)
    Pob_2=np.zeros((Nr*ne,LC),dtype=int).tolist()
    for i in range(Nr*ne):
        for j in range(LC):
                Pob_2[i][j]=np.random.randint(2)
    Pob_10= fPobDec(Pob_2,ne,Nr)
    Pob_float = fScale(Pob_10,lim,ne,LC,Nr,ndecimals=ndecimals4float)
    Pob_2=[]
    Pob_10=[]
    SpecialCrossoverOperator='SBX'
    print("SpecialCrossoverOperator='SBX'")
    
    #Initialize parameters
    Nr00=Nr
    Nrc=0
    Pob_floatc=[]
    fxc=[]
    fxpc=[]
    punishc=[]
    Pob_2c=[]
    Nraux=Nr
    lenfront0=1
    change=0
    front=[[]]
    lenindc2=0
    Nrelite=NumIndiv
    stop=0
    indivcompiting=2
    tstart = time.time()
    
    T=max(np.ceil(0.15*Nr),2)
    T=int(min(max(T,2),15))
    fx,breaking=ObjSpace(Pob_float)
    z=np.min(fx,axis=0)
    zN=np.max(fx,axis=0)
    eps=1e-15
    theta=5*max(lim[il][1] for il in range(len(lim)))
    sp_lambda,sp_neighbors=CreateSubProblems(multiobjectives,Nr,T)
    normlambda=np.zeros((1,Nr))[0]
    for fil in range(Nr):
        normlambda[fil]=np.linalg.norm(sp_lambda[fil])
    gNr=np.zeros((1,Nr))[0]
    alphaIMOEAD=0.2; betaIMOEAD=0.8
    for fil in range(Nr):
        fxNr=np.array(fx)[fil]
        fxNrprime=fxNr.copy()-z
        gNr[fil]=alphaIMOEAD*np.sum(sp_lambda[fil]*np.abs(np.array(fxNr)))+betaIMOEAD*np.max(sp_lambda[fil]*np.abs(np.array(fxNrprime)))
    _,front=ffastnondominatedsortELY(Nr,multiobjectives,fx,ne,Pob_float,Nr-1)
    fxc,Pob_floatc=Objects(Nr,Nrc,ne,multiobjectives,front[0],fx,Pob_float)
    
    #Main loop
    contG=0
    while contG<NGenerations:
        if contG>1 and not isinstance(Nstopelite,bool) and stop>=Nstopelite:
            print('\rstop')
            break
        fx,breaking=ObjSpace(Pob_float)
        if breaking==True:
            print('fx error',fx)
            break
        contG+=1
        
        
        for fil in range(Nr):
            KM=np.random.randint(0,T-1,2)
            j1=sp_neighbors[fil][KM[0]]
            j2=sp_neighbors[fil][KM[1]]
            if binary:
                Pob_10=fScalInv(Pob_float[j1*ne:(j1+1)*ne],lim,ne,LC,1);Pob_2p1=fPobDecInv(Pob_10,ne,1,LC)
                Pob_10=fScalInv(Pob_float[j2*ne:(j2+1)*ne],lim,ne,LC,1);Pob_2p2=fPobDecInv(Pob_10,ne,1,LC)
                Pob_2y2,Parents,Couples=freproduction( Pob_2p1+Pob_2p2,2,ne,[1,1],LC,SpecialCrossoverOperator=SpecialCrossoverOperator)
                Pob_2y=Pob_2y2[:ne]
                Pob_2y=fMutation(Pob_2y,LC,MutationPorc*2)#
                Pob_10y= fPobDec(Pob_2y,ne,1);Pob_floaty = fScale(Pob_10y,lim,ne,LC,1,ndecimals=ndecimals4float)
            else:
                Pob_floaty1=Pob_float[j1*ne:(j1+1)*ne]
                Pob_floaty2=Pob_float[j2*ne:(j2+1)*ne]
                Pob_floaty0=Crossover_mutationNoBinary(Pob_floaty1+Pob_floaty2,[1,1],MutationPorc,ne,2,lim,onlycrossover=True)
                if np.random.randint(2):Pob_floaty=Pob_floaty0[ne:]#((np.array(Pob_floaty0)[ne:]+np.array(Pob_floaty0)[:ne])/2).tolist()#
                else:Pob_floaty=Pob_floaty0[:ne]
            
            fxy,breaking=ObjSpace(Pob_floaty)
            if breaking==True:break
            z=np.min([z.tolist(),fxy[0]],axis=0)
            fxprime=fxy[0].copy()-z
            for fill in sp_neighbors[fil]:
                gy=alphaIMOEAD*np.sum(sp_lambda[fil]*np.abs(np.array(fxy[0])))+betaIMOEAD*np.max(sp_lambda[fil]*np.abs(np.array(fxy[0])-z))
                if gy<=gNr[fill]:
                    Pob_float[fill*ne:(fill+1)*ne]=Pob_floaty
                    gNr[fill]=gy
                    fx[fill]=fxy[0].copy()
        sp_lambda+= np.random.normal(0, 1/Nr, size=sp_lambda.shape)
        sp_lambda = sp_lambda / np.sum(sp_lambda, axis=1, keepdims=True)
        
        
        if considerfront or (contG>=NGenerations):#stop>=Nstopelite-1 or 
            _,front=ffastnondominatedsortELY(Nr,multiobjectives,fx,ne,Pob_float,2)
            zN=np.max(np.array(fx)[front[0]],axis=0)
            # front=[np.arange(Nr).tolist()]
            whererank1=front[0]
            minfx=[[0 for ill in range(multiobjectives)] for il in range(len(whererank1))]
            Pob_floatmax=[[0 for il in range(ne)] for ill in range(len(whererank1))]
            for fil in range(len(whererank1)):
                indmax=whererank1[fil]
                ri=(indmax)*ne;rj=(indmax+1)*ne
                Pob_floatmax[fil][:]=copy.deepcopy(Pob_float[ri:rj])
                minfx[fil][:]=copy.deepcopy(fx[indmax])
        lenunique=len(front[0])
        
        
        if contG==1 or contG%printcont==0:
            TEXT='\r'
            TEXT+='len(front[0])='+str(len(front[0]))
            TEXT+='unique='+str(lenunique)
            TEXT+=' EndGeneration='+str(contG)+'                     '
            sys.stdout.write(TEXT)
            sys.stdout.flush()
        
        
        if lenunique<=lenfront0:
            stop+=1
        else:
            stop=0
        lenfront0=max(lenfront0,lenunique)
        
    runtime = time.time() - tstart
    print('\n elapsed [s]=',runtime)
    print('Pob_floatmax[0]=',Pob_floatmax[0])
    return Pob_floatmax,minfx,runtime