# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:39:10 2024

@author: Edgar Mora
"""
import numpy as np
import MainModelingAdvELY as MMAE
import matplotlib.pyplot as plt
import Problems
#%% Input data
example='DTLZ1'# This can be switched between DTLZ1 to DTLZ7, ZDT1 to ZDT6, WFG1 to WFG9, and MMF1 to MMF8.
algorithm='ELY'# This can be switched among 'ELY', 'NSGAII', 'SDNSGAII', and 'IMOEAD'
Nstopelite=False
Npopulation=800
mutation=10
Ngeneraciones=100
#%% Problem limits according to papers
if example[:-1]=='DTLZ':
    multiobjectives=3
    dimensions=3
    lim_dimensions=[[0,1,'float'] for _ in range(dimensions)]
elif example[:-1]=='ZDT':
    multiobjectives=2
    dimensions=2
    lim_dimensions=[[0,1,'float'] for _ in range(dimensions)]
elif example[:-1]=='WFG':
    multiobjectives=3
    dimensions=10
    if example=='WFG3':
        lim_dimensions=[[0,3,'float'] for _ in range(dimensions)]
    elif  not (algorithm=='ELY'):#other algorithms provide outliers points. These limits have been changed for visualization proposes.
        lim_dimensions=[[0,3.5,'float'] for _ in range(dimensions)]
    else:
        lim_dimensions=[[0,10,'float'] for _ in range(dimensions)]
elif example[:-1]=='MMF':
    multiobjectives=2
    dimensions=2
    if example=='MMF1':lim_dimensions=[[0,3,'float'],[-1,1,'float']]
    elif example=='MMF2':lim_dimensions=[[0,1,'float'],[0,2,'float']]
    elif example=='MMF3':lim_dimensions=[[0,1,'float'],[0,1.5,'float']]
    elif example=='MMF4':lim_dimensions=[[-1,1,'float'],[0,2,'float']]
    elif example=='MMF5':lim_dimensions=[[1,3,'float'],[-1,3,'float']]
    elif example=='MMF6':lim_dimensions=[[1,3,'float'],[-1,2,'float']]
    elif example=='MMF7':lim_dimensions=[[1,3,'float'],[-1,1,'float']]
    elif example=='MMF8':lim_dimensions=[[-np.pi,np.pi,'float'],[0,9,'float']]
#%% Multi-objective optimization algorithms
if algorithm=='ELY':
    Pob_floatmax,minfx,runtime=MMAE.ELY('Problems.'+example,lim_dimensions,
                       Npopulation,mutation,Ngeneraciones,Nstopelite=Nstopelite,multiobjectives=multiobjectives,epsilon_bar=1e-10,binary=False)
    # The results for MMF problems are better when binary=True
elif algorithm=='NSGAII':
    Pob_floatmax,minfx,runtime=MMAE.NSGAII('Problems.'+example,lim_dimensions,
                       Npopulation,mutation,Ngeneraciones,Nstopelite=Nstopelite,multiobjectives=multiobjectives,binary=False)
    # The results for MMF problems are better when binary=True
elif algorithm=='SDNSGAII':
    if example[:-1]=='MMF':binary=True #This works much better with binary coding operators
    else:binary=False
    Pob_floatmax,minfx,runtime=MMAE.SDNSGAII('Problems.'+example,lim_dimensions,
                       Npopulation,mutation,Ngeneraciones,Nstopelite=Nstopelite,multiobjectives=multiobjectives,binary=binary)
elif algorithm=='IMOEAD':
    Pob_floatmax,minfx,runtime=MMAE.IMOEAD('Problems.'+example,lim_dimensions,
                       Npopulation,mutation,Ngeneraciones,Nstopelite=Nstopelite,multiobjectives=multiobjectives,binary=False)
#%% Draw results
print(Pob_floatmax[0])
Pob_floatmax=np.unique(Pob_floatmax,axis=0)
Pob_floatmax0=np.copy(Pob_floatmax);minfx0=np.copy(minfx)
drawresults=True
if drawresults:
     if dimensions>2:
         ax2 = plt.figure(1).add_subplot(projection='3d')
     else:
         ax2 = plt.figure(1).add_subplot()
     for i in range(len(Pob_floatmax)):
         x1=Pob_floatmax[i][0]; x2=Pob_floatmax[i][1]; 
         if dimensions>2:
             x3=Pob_floatmax[i][2]
             ax2.plot(x1,x2,x3,'x',color='red')
         else:
             x3=0
             ax2.plot(x1,x2,'x',color='red')
     ax2.set_ylabel('x1');ax2.set_xlabel('x2');
     if dimensions>2:ax2.set_zlabel('x3')
     if multiobjectives>2:
         ax3 = plt.figure(2).add_subplot(projection='3d')
     else:
         ax3 = plt.figure(2).add_subplot()
     for i in range(len(Pob_floatmax)):
         f,_=eval('Problems.'+example+'(Pob_floatmax[i])')
         f1=f[0];f2=f[1];
         if multiobjectives>2:
             f3=f[2]
             ax3.plot(f1,f2,f3,'.',color='darkorange')
         else:
             f3=0
             ax3.plot(f1,f2,'.',color='darkorange')
     if multiobjectives>2:
         ax3.grid(False)
         ax3.xaxis.pane.fill = False # Left pane
         ax3.yaxis.pane.fill = False # Right pane
         ax3.xaxis.pane.set_edgecolor('black')
         ax3.yaxis.pane.set_edgecolor('black')
         ax3.zaxis.pane.set_edgecolor('black')
         ax3.zaxis.pane.set_facecolor("white")
         ax3.view_init(azim=45,elev=25)
     ax3.set_ylabel('$f_{1}$'+'(x)',{'fontname':'Cambria'});
     ax3.set_xlabel('$f_{2}$'+'(x)',{'fontname':'Cambria'});
     if multiobjectives>2:ax3.set_zlabel('$f_{3}$'+'(x)',{'fontname':'Cambria'})

     ax3.set_title(example,{'fontname':'Cambria'})
