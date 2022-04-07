from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import preprocessing
from qpth.qp import QPFunction, QPSolvers
from tqdm import tqdm
#pip install cvxpy
import warnings
warnings.filterwarnings("ignore")

class nn_controller(nn.Module):
    def __init__(self):
        super(nn_controller, self).__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50,100)
        self.fc3 = nn.Linear(50,1)

    def forward(self, x):
        x = self.fc3(self.fc2(self.fc1(x)))
        return x
    
class barrier_controller(nn.Module):
    def __init__(self):
        super(barrier_controller, self).__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50,100)
        self.fc3 = nn.Linear(100,5)
        self.Q = Variable(0.5*torch.eye(1).cuda(),requires_grad=True)
        self.p1 = torch.Tensor().cuda()
        self.p2 = torch.Tensor().cuda()
        self.p3 = torch.Tensor().cuda()
        self.p4 = torch.Tensor().cuda()
        self.G = torch.ones(1,3).cuda()
        self.h = torch.ones(1,3).cuda()
        self.e = Variable(torch.Tensor())
    
    def CBF_Parameters(self,X,alpha1,alpha2,alpha3,s_star,v_star,tilde_vh):
        X=X.reshape(6,-1)
        A=torch.tensor([]).cuda()
        b=torch.tensor([]).cuda()
        state_size=X.shape
        f=torch.zeros((state_size[0],state_size[0])).cuda()
        tau=4
        #assign fx
        for i in range(1,int(state_size[0]/2-1)+1):
            f[2*i,2*i-1]=1
            f[2*i,2*i+1]=-1
            f[2*i+1,2*i-1]=alpha3
            f[2*i+1,2*i]=alpha1
            f[2*i+1,2*i+1]=-alpha2

        #assign gx
        g=torch.zeros(state_size[0]).cuda()
        g[1]=1

        for index in range(2,int(state_size[0]/2)+1):
            s_safe=tau*(X[index*2-1]-X[index*2-3])
            delta_b=torch.zeros(state_size[0]).cuda()
            delta_b[index*2-2]=1
            delta_b[index*2-1]=-tau
            delta_b[index*2-3]=tau
            eta=X[index*2-2]+s_star-s_safe
            
            while delta_b[1]==0:
                delta_b=(delta_b.unsqueeze(0)).mm(f).squeeze(0)
                eta=torch.cat((eta.unsqueeze(0),(delta_b.unsqueeze(0).mm(X))),0)
                
            A=torch.cat((A,-delta_b[1].unsqueeze(0)),0)
            if index==2:
                #K=self.p1
                b=delta_b.unsqueeze(0).mm(f.mm(X))+self.p1.mul(eta)
                
            elif index==3:
                K=torch.concat((self.p2,self.p3),0)
                temp=K.mul(eta)
                ans=temp[0,:]+temp[1,:]
                new_b = (delta_b.unsqueeze(0)).mm(f.mm(X))+ans
                b=torch.cat((b,new_b),0)
                

        #for disturbance from head vehicle
        '''
        if tilde_vh!=None:
            tau=torch.tensor([3.5]).cuda()
            h=X[0,:]+s_star-tau*X[1,:]+tau*tilde_vh#;%-tilde_vh
            A=torch.cat((A,tau),0)#;%tau
            coef=12
            new_b=X[1]-tau*(alpha1*X[0]-alpha2*X[1])+h*coef*self.p4
            b=torch.cat((b,new_b),0)
        '''
        self.h=b#.permute(1,0)
        self.G=A.unsqueeze(1).expand(self.h.shape)
        self.h=self.h.permute(1,0)
        self.G=self.G.permute(1,0).unsqueeze(2)
    
    def forward(self,z,tilde_vh,alpha1=1.2566,alpha2=1.5,alpha3=0.9,s_star=20,v_star=20):
        u = self.fc3(self.fc2(self.fc1(z)))
        self.p1=u[:,1].unsqueeze(0)
        self.p2=u[:,2].unsqueeze(0)
        self.p3=u[:,3].unsqueeze(0)
        self.p4=u[:,4].unsqueeze(0)
        self.CBF_Parameters(z,alpha1,alpha2,alpha3,s_star,v_star,tilde_vh)
        self.Q=Variable(0.5*torch.ones(u[:,0].shape[0],1,1).cuda(),requires_grad=True)
        u_ = QPFunction()(self.Q.float(), -u[:,0].unsqueeze(1).float(), self.G.float(), self.h.float(),self.e,self.e).float()
        
        return u_
    
def train_nn_controller(model):
    data=loadmat('dataset/data.mat')
    dataset = data['State']
    
    v_star = 20
    s_star = 20
    

    V_diff = dataset[:,0:-1,1] - dataset[:,1:,1]
    D_diff = dataset[:,0:-1,0] - dataset[:,1:,0]
    
    X = np.zeros((len(dataset),6))
    X[:,0::2]=D_diff-s_star
    X[:,1::2]=dataset[:,1:,1]-v_star

    portion=0.3
    train = X[:int(portion*len(X)),:]
    test = X[int(portion*len(X)):,:]

    train_label = dataset[:int(portion*len(X)),1,[2]]
    test_label = dataset[int(portion*len(X)):,1,[2]]

    minmax_scale = preprocessing.MinMaxScaler().fit(X)
    train = minmax_scale.transform(train)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data=Variable(torch.tensor(train).to(torch.float32))
    train_label=Variable(torch.tensor(train_label).to(torch.float32))
    test_data=Variable(torch.tensor(test).to(torch.float32))
    test_label=Variable(torch.tensor(test_label).to(torch.float32))

    model = model.to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    epochs = 3000
    epoch_disp = 300
    for epoch in range(epochs):
        outputs = model(train_data.to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, train_label.to(device))
        loss.backward()
        optimizer.step()

        if epoch % epoch_disp == 0:
              print("Epoch: %d, loss: %1.10f" % (epoch, loss.item()))
    
    return model,minmax_scale


def train_barriernet_controller(model):
    data=loadmat('dataset/data.mat')
    dataset = data['State']
    
    v_star = 20
    s_star = 20
    

    V_diff = dataset[:,0:-1,1] - dataset[:,1:,1]
    D_diff = dataset[:,0:-1,0] - dataset[:,1:,0]
    
    X = np.zeros((len(dataset),6))
    X[:,0::2]=D_diff-s_star
    X[:,1::2]=dataset[:,1:,1]-v_star

    portion=0.7
    train = X[int(portion*len(X)):,:]
    test = X[:int(portion*len(X)),:]

    train_label = dataset[int(portion*len(X)):,1,2].reshape(-1,1)
    test_label = dataset[:int(portion*len(X)),1,2].reshape(-1,1)

    #minmax_scale = preprocessing.MinMaxScaler().fit(X)
    #train = minmax_scale.transform(train)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data=Variable(torch.tensor(train).to(torch.float32))
    train_label=Variable(torch.tensor(train_label).to(torch.float32))
    test_data=Variable(torch.tensor(test).to(torch.float32))
    test_label=Variable(torch.tensor(test_label).to(torch.float32))

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    epochs = 801
    epoch_disp = 1
    for epoch in tqdm(range(epochs)):
        outputs = model(train_data.to(device),torch.tensor(dataset[:int(portion*len(X)),0,1]-v_star).to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, train_label.to(device))
        loss.backward()
        optimizer.step()

        if epoch % epoch_disp == 0:
            print("Epoch: %d, loss: %1.10f" % (epoch, loss.item()))
    torch.save(model.state_dict(), "barriernet.pkl")
    
    return model,None#minmax_scale


def load_barriernet_controller(model):
    data=loadmat('dataset/data.mat')
    dataset = data['State']
    
    v_star = 20
    s_star = 20
    
    V_diff = dataset[:,0:-1,1] - dataset[:,1:,1]
    D_diff = dataset[:,0:-1,0] - dataset[:,1:,0]
    
    X = np.zeros((len(dataset),6))
    X[:,0::2]=D_diff-s_star
    X[:,1::2]=dataset[:,1:,1]-v_star

    portion=0.2
    train = X[:int(portion*len(X)),:]
    test = X[int(portion*len(X)):,:]

    train_label = dataset[:int(portion*len(X)),1,2].reshape(-1,1)
    test_label = dataset[int(portion*len(X)):,1,2].reshape(-1,1)

    minmax_scale = preprocessing.MinMaxScaler().fit(X)
    
    model.load_state_dict(torch.load("barriernet.pkl"))
    
    return model,minmax_scale