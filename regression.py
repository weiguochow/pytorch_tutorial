import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)

y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x),Variable(y)

plt.scatter(x.data.numpy(),y.data.numpy())

plt.show()

import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_output):
        super(Net,self).__init__()
        
        self.hidden1 = torch.nn.Linear(n_feature,n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1,n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2,n_output)
    
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x
    
    
net = Net(n_feature=1,n_hidden1=10,n_hidden2=10,n_output=1)

print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

loss_func = torch.nn.MSELoss()


plt.ion()

plt.show()


max_iteration = 1000

for i in range(max_iteration):
    prediction = net(x)
    
    loss = loss_func(prediction,y)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
    print('Step: %4d, Loss is %.4f'%(i ,loss.data[0]))
    
    if i%(max_iteration-1) == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5, 0, 'Loss=%.4f,step=%4d' % (loss.data[0],i), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)


x_test_tensor = torch.FloatTensor([0.75])

x_test_tensor = Variable(x_test_tensor)

y_test = net(x_test_tensor)

print(y_test.data.numpy())
