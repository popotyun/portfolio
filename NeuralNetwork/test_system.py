import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import itertools

from my_function import my_func1 as mf #自作モジュール　

from scipy.stats import norm

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

input_dim=0
Epsilon = 0.1







class cell():

    def __init__(self,index,neiborhood_index,input_dim,state,next_state,q,action,reward,old_state):

        self.index = index
        self.neiborhood_index=neiborhood_index
        self.input_dim=input_dim
        self.state=state
        self.next_state=next_state
        self.q=q
        self.action=action
        self.reward=reward
        self.old_state=old_state

def select_action(q):
    r = np.random.rand()
    if r < Epsilon:
        am=np.random.randint(3)
        return am

    else:
        am = 0
        m = q[am]
        for i in range(len(q)-1):
            if m < q[i+1]:
                m = q[i+1]
                am = i+1
        return am

def gauss(x, a=1, mu=0, sigma=1):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

"""
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
"""

def custom_loss(output,q,reward,gamma):
    loss=(reward+(gamma*q)-output)*(reward+(gamma*q)-output)

    return loss



def update_q(reward,q,old_q): #Update Q-value (required by the learning algorithm: RbLA)
    #Q[olds][olda] += Alfa*(reward+Gamma*Q[s][a]-Q[olds][olda])

    old_q+= alfa *(reward+gamma*q-old_q)
    #Alfa*(reward+Gamma*Q[s][a]-Q[olds][olda])
    return old_q





action = 3 #0.1度増減+そのまま
state = 1000 #状態数 1000時間？
alfa = 0.1 # 学習係数
epsilon = 0.1 # 貪欲法
gamma = 0.95 # 割引率




"""
Q = np.random.rand(state,action)#各セルのQテーブル

#各セルにQテーブルを格納

array=[]
cells=[]
loop=np.zeros((5,5))
for i in range(len(loop)):
    array=[]
    for j in range(len(loop[i])):

        #array.append(Q)
        cells.append(Q)

cells=np.array(cells)
print(len(cells))
"""

Q=np.random.rand(1,3)


loop=np.zeros((5,5))
cells=[]
"""
for i in range(len(loop)):
    array=[]
    for j in range(len(loop[i])):

        #array.append(Q)
        cells=np.append(cells,Q)

cells=np.array(cells)
print(len(cells))
"""


s=np.zeros((5,5))#状態




file="area1.csv"
x_train=mf.dataload(file,"2018-12-31 00:00:00")
file="area2.csv"
test_data=mf.dataload(file,"2020-12-31 00:00:00")
nei_index=[]
net=[]
"""
for i in range(len(loop)):
    for j in range(len(loop[i])):
        a=mf.moore_neighborhood(j,i)
        nei_index.append(a)
        input_dim=len(a)
        net.append(Net())


for i in range(len(net)):
    # set training mode
    net[i].train()

# set training parameters
optimizer = torch.optim.SGD(net[i].parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
"""
measurable_cells=[[0,0],[4,4]]
predict_cells=[[0,0],[4,4]]
for k in range(4):
    for i in range(len(predict_cells)):
        nei = mf.moore_neighborhood(predict_cells[i][1],predict_cells[i][0])
        for j in nei:
            predict_cells.append(j)


predict_cells=mf.get_duplicate_list_order(predict_cells)
print(predict_cells,len(predict_cells))

class_cells=np.full((5,5),-1,dtype='float128')

models=[]
"""
for i in range(len(class_cells)):

    for j in range(len(class_cells[i])):

        index=mf.moore_neighborhood(j,i)
        var_array=[]
        error_array=[]
        c = cell([i,j],index,len(index),-1,-1,error_array,error_array,error_array,-1)
        cells.append(c)
        model = Sequential()
        #input
        n_in=len(index)
        n_mid1=8
        n_mid2=8
        n_out=3
        model.add(Dense(n_mid1, input_shape=(n_in,), activation='relu'))
        model.add(Dense(n_mid2, activation='relu'))
        model.add(Dense(n_out, activation='relu')) # output
        #学習方法　学習スタイル　目的関数　評価手法

        model.compile(loss="mean_squared_error", optimizer="Adam",
             metrics=['accuracy'])
        print(model.summary())
        models.append(model)
print(len(cells))
"""
nn=Sequential()

nn.add(Dense(256, input_shape=(2,), activation='relu'))
nn.add(Dense(256, activation='relu'))
nn.add(Dense(25, activation='relu'))
nn.compile(loss="mean_squared_error", optimizer="Adam",
     metrics=['accuracy'])

y_train=x_train
#正規化処理　最小-30 最大50
#入力の式:(float(y_train[mes_index][t])+30)/80
#出力の式:(float((80*出力値)-30)
input_data=[]
label_data=[]
print(len(x_train))
a1=x_train[1][:]
a2=x_train[25][:]

print(a1)
print(a2)
test_a1=test_data[1][:]
test_a2=test_data[25][:]
test_input_data=[]

for i in range(0,len(a1)-2):
    data=(float(a1[i])+30)/80
    data2=(float(a2[i])+30)/80
    input_data.append([float(data),float(data2)])

for i in range(len(test_a1)-1):
    data=(float(test_a1[i])+30)/80
    data2=(float(test_a2[i])+30)/80
    test_input_data.append([float(data),float(data2)])

print(input_data[len(input_data)-1],len(input_data))

for i in range(1,len(a1)-1):
    label=[]
    for j in range(25):
        data=(float(x_train[j+1][i+1])+30)/80
        label.append(float(data))


    label_data.append(label)

print(label_data[len(label_data)-1],len(label_data))
#for i in input_data:
nn.fit(np.array(input_data),np.array(label_data),epochs=1000)


name="test_system.h5"
nn.save(name)
ax=[]
Figure=plt.figure()
for i in range(25):
    ax.append(Figure.add_subplot(5,5,i+1))


plot=[]
for i in range(30):
    for j in nn.predict(np.array([test_input_data[i]])):

        plot.append(j)
    cnt=0
print(plot)
plot=np.array(plot)
plot=plot.T
print(len(plot))
array=[]
array2=[]
print(plot)
print(plot[0][1])

for i in range(len(plot)):
    array=[]
    for j in range(len(plot[i])):
            #print((float((80*plot[i][j])-30)))
        array.append((float((80*plot[i][j])-30)))
    array2.append(array)
cnt=0
print(array2)
plot=array2
print(len(plot))
for i in plot:
    ax[cnt].plot(i,label="predict")
    ax[cnt].legend()
    cnt+=1

cnt=0
array=[]
array2=[]

for i in range(len(test_data)):
    array=[]
    for j in range(30):
        if i!=0:
            #for k in x_train[j+1]:
            array.append(float(test_data[i][j+1]))

    if(i!=0):
        array2.append(array)
            #plot_data=float(x_train[j][i])
            #if cnt !=0:
plot_data=array2
cnt=0
print(len(plot_data))
for i in plot_data:

    ax[cnt].plot(i,label="label")
    ax[cnt].legend()
    cnt+=1
plt.show()

#for i in input_data:
    #print(i,nn.predict(np.array([i])))




























