import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from fc_controller import *
import torch
import torch.nn as nn
import pandas as pd 
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def CBF_Control(X,u_ref,alpha1,alpha2,alpha3,s_star,v_star,tilde_vh):
    A=[]
    b=[]
    state_size=np.shape(X)
    f=np.zeros((state_size[0],state_size[0]))
    tau=4
    #assign fx
    for i in range(1,int(state_size[0]/2-1)+1):
        f[2*i,2*i-1]=1
        f[2*i,2*i+1]=-1
        f[2*i+1,2*i-1]=alpha3
        f[2*i+1,2*i]=alpha1
        f[2*i+1,2*i+1]=-alpha2

    #assign gx
    g=np.zeros(state_size[0])
    g[1]=1

    for index in range(2,int(state_size[0]/2)+1):
        s_safe=tau*(X[index*2-1]-X[index*2-3])
        delta_b=np.zeros(state_size[0])
        delta_b[index*2-2]=1
        delta_b[index*2-1]=-tau
        delta_b[index*2-3]=tau
        eta=[X[index*2-2]+s_star-s_safe]
        
        while delta_b[1]==0:
            delta_b=delta_b.dot(f)
            eta.append(delta_b.dot(X))
             
        A.append(-delta_b[1])
        
        #poles=[1000,5,4.9,4.8]
        #K=fliplr(poly(poles))
        if index==2:
            K=np.array([1])
        elif index==3:
            K=np.array([1,0.85]) #ones(1,length(eta)+1);
        eta=np.array(eta)
        b.append(float(delta_b.dot(f.dot(X))+K.dot(eta.reshape(-1,1))))#;%(1:length(K)-1)


    #for disturbance from head vehicle
    if tilde_vh!=None:
        tau=3.5
        h=X[0]+s_star-tau*X[1]+tau*tilde_vh#;%-tilde_vh
        A.append(tau)#;%tau
        coef=12
        b.append(-float(X[1])-tau*(alpha1*float(X[0])-alpha2*float(X[1]))+coef*float(h))
        
    A=np.array(A).reshape(-1,1)
    b=np.array(b).reshape(-1,1)
    H=np.array([1])
    f_ = np.array([-u_ref])
    u=solve_qp(H,f_,A,b)

    if u == None:
        u=b[len(b)-1]/A[len(A)-1]
        
    return u

def empty_filter(X,u_ref,alpha1,alpha2,alpha3,s_star,v_star,tilde_vh):
    return u_ref

def leading_cruise_control_modeling(safety_filter = empty_filter, m = 0, n = 2, PerturbedID = 0, PerturbedTime = 1,PerturbedAccel = -4, PerturbedType = 2, TotalTime = 20, Tstep = 0.01, controller = None, scale= None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Parameters in the car-following model
    alpha = 0.6 #Driver Model: OVM
    beta  = 0.9
    s_st  = 5
    s_go  = 35
    
    #Traffic equilibrium
    v_star   = 20 #Equilibrium velocity
    acel_max = 7
    dcel_max = -7
    v_max    = 40
    s_star   = np.arccos(1-v_star/v_max*2)/np.pi*(s_go-s_st)+s_st #Equilibrium spacing
    #linearized model
    alpha1 = alpha*v_max/2*np.pi/(s_go-s_st)*np.sin(np.pi*(s_star-s_st)/(s_go-s_st))
    alpha2 = alpha+beta
    alpha3 = beta
    #Simulation length
    NumStep   = TotalTime/Tstep

    #Initial state definition
    S = np.zeros((int(NumStep),m+n+2,3))
    dev_s = 0
    dev_v = 0
    co_v  = 1.0
    v_ini = co_v*v_star #Initial velocity
    # from - dev to dev
    S[0,:,[0]] = np.linspace(0,-(m+n+1)*s_star,m+n+2) + (rand(1,m+n+2)*2*dev_s - dev_s)
    #The vehicles are uniformly distributed on the straight road with a random deviation
    S[0,:,[1]] = v_ini * np.ones((1,m+n+2)) + (rand(1,m+n+2)*2*dev_v-dev_v)

    #Perturb ID
    ID = np.zeros(m+n+2)
    ID[m+1] = 1

    X = np.zeros((2*(m+n+1),int(NumStep)))
    u = np.zeros((int(NumStep),1))               # 0. HDV  1. CAV
    V_diff = np.zeros((int(NumStep),m+n+1))      # Velocity Difference
    D_diff = np.zeros((int(NumStep),m+n+1))      # Following Distance

    #Feedback gain
    K = np.zeros(2*(n+1))
    K[0:6] = np.array([0.2,-0.5,-0.2,0.05,-0.1,0.05])

    #Simulation begin
    for k in range(int(NumStep)):
        #Update acceleration
        V_diff[k,:] = S[k,0:-1,1] - S[k,1:,1]
        D_diff[k,:] = S[k,0:-1,0] - S[k,1:,0]
        
        cal_D = D_diff[k,:] # For the boundary of Optimal Veloicity Calculation
        for i in range(m+n+1):
            if cal_D[i] > s_go:
                cal_D[i] = s_go
            elif cal_D[i] < s_st:
                cal_D[i] = s_st
        #nonlinear OVM Model
        acel = alpha*(v_max/2*(1-np.cos(np.pi*(cal_D-s_st)/(s_go-s_st))) - S[k,1:,1])+beta*V_diff[k,:]
        acel[acel>acel_max] = acel_max
        acel[acel<dcel_max] = dcel_max
        S[k,1:,2] = acel
        S[k,0,2] = 0 #the preceding vehicle

        if PerturbedType == 1: #sine wave
            P_A = 0.2
            P_T = PerturbedTime
            if k*Tstep>0 and k*Tstep<0+P_T:
                S[k,PerturbedID,2] = P_A*np.cos(2*np.pi/P_T*(k*Tstep-20))

        elif PerturbedType == 2: #brake/accel
            if k*Tstep>0 and k*Tstep<=0+PerturbedTime:
                S[k,PerturbedID,2] = PerturbedAccel
            #if k*Tstep>=0+PerturbedTime and k*Tstep<0+2*PerturbedTime:
                #S[k,PerturbedID,2] = -PerturbedAccel

        X[0::2,[k]] = D_diff[k,:].reshape(m+n+1,1) - s_star
        X[1::2,[k]] = S[k,1:,1].reshape(m+n+1,1) - v_star
        
        #nominal controller + safety filter
        if controller == None:
            u[k] = K.dot(X[:,k])
        else:
            controller.eval()
            predict_data = X[:,[k]].reshape(1,-1)#scale.transform()
            
            u[k] = controller(torch.tensor(predict_data).to(device).to(torch.float32),torch.tensor(S[k,0,1] -v_star).to(device).to(torch.float32)).detach().cpu().numpy()
            
        u[k] = safety_filter(X[:,k],u[k],alpha1,alpha2,alpha3,v_star,s_star,S[k,0,1] -v_star)
        if u[k] > acel_max:
            u[k] = acel_max
        elif u[k] < dcel_max:
            u[k] = dcel_max
        S[k,m+1,2] = u[k]
        
        
        if k!=NumStep-1:
            S[k+1,:,1] = S[k,:,1] + Tstep*S[k,:,2]
            S[k+1,:,0] = S[k,:,0] + Tstep*S[k,:,1]
    return S

if __name__ == '__main__':
    fc,scale = train_barriernet_controller(barrier_controller().cuda())
    states = leading_cruise_control_modeling(controller = fc, TotalTime = 10, PerturbedTime = 1.5,PerturbedAccel = 7,PerturbedID = 3,scale=scale)
    np.save(file="disp/states_nn.npy", arr=states)
    
    