import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import itertools

from my_function import my_func1 as mf #自作モジュール　

def gauss(x, a=1, mu=0, sigma=1):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))


def selectAction(s): #selecting action by epsilon greedy method (required by the learning algorithm: RbLA)
    #state_index=s[0]*16**0
    """
    state_index=0
    for i in range(len(s)):
        state_index+=s[i]*(16**i)
    """
    r = np.random.rand()
    if r < Epsilon:
        am=np.random.randint(ActionN)
        return am
    else:
        am = 0
        m = Qtable[s][am]
        for i in range(1, ActionN):
            if m < Qtable[s][i]:
                m = Qtable[s][i]
                am = i
        return am

def update_q(s, olds, a, olda, reward): #Update Q-value (required by the learning algorithm: RbLA)
    #print(s,olds)
    """
    old_state_index=0
    state_index=0
    for i in range(len(olds)):
        old_state_index+=olds[i]*(16**i)
    for i in range(len(s)):
        state_index+=s[i]*(16**i)
    """
    Qtable[olds][olda] += Alfa*(reward+Gamma*Qtable[s][a]-Qtable[olds][olda])
    #print(Qtable[olds][:])
"""
Qテーブルは，5度刻みで状態とする．
-30から50なので，80/5=16
16**周辺の状態（4~9)*3(行動数)
16**3*3
"""
# -*- coding: utf-8 -*-
#simple maze
#implementation of Sarsa

def get_float(array):
    array2=[]
    for i in array:
        array2.append(float(i))

def get_state(bins,num):
    bins[0]=-999999
    bins[len(bins)-1]=999999
    for i in range(len(bins)-1):
        if bins[i] <= num and num < bins[i+1]:
            return i





file="/Users/shimizutakumi/Documents/PythonWorks-4/mesh_data/area1.csv"
x_train=mf.dataload(file,"2018-12-31 00:00:00")
index=mf.moore_neighborhood(0,0)
print(index)
a=[]
#l_n_str = [str(n) for n in l_n]
input1=x_train[2][:]
input2=x_train[6][:]
input3=x_train[7][:]
input4=x_train[1][:]
input1=[float(n) for n in input1]
input2=[float(n) for n in input2]
input3=[float(n) for n in input3]
input4=[float(n) for n in input4]

max=max(input1[0:1440])
min=min(input1[0:1440])
diif=(max-min)/16
print(min,max,max-min)
print((max-min)/16)
"""
1.425
min~min+1.425=0,min+1.425~min+1.425*2=2,.......~max=16
input1*16**0+input2*16**1....=Q_index

"""
s_cut, bins = pd.cut(input1[0:1440], 16, retbins=True)
print(bins)#境界値




StateN=16**4
print(StateN)
ActionN = 3
Qtable=np.random.rand(StateN+1,ActionN)
print(len(Qtable))
state_index=0
for j in range(4):
    state_index+=15*(16**j)
print(state_index)

EpiNum = 200 #number of episode
Alfa = 0.01 # learning coefficient
Epsilon = 0.1 # Coefficient for Epsilon-greedy method
Gamma = 0.95 # Discount rate
Agent_state=input4[0]#初期値
EpiNum=200
temp=(get_state(bins,input4[0])*1.425)-16.0
print(len(Qtable))
#print(Qtable[65536][:])
for i in range(100):
    #現在の周囲の気温を離散化して取得
    state=[]
    state.append(get_state(bins,input4[i]))#自身の現在の気温
    #周辺の気温
    state.append(get_state(bins,input1[i]))
    state.append(get_state(bins,input2[i]))
    state.append(get_state(bins,input3[i]))

    print(state,input1[i],input2[i],input3[i],input4[i],get_state(bins,input4[i+1]))

    for cnt in range(EpiNum):
        #初期化
        state_index=0
        for j in range(len(state)):
            state_index+=state[j]*(16**j)
        #print
        s=state_index
        a=selectAction(s)
        #temp=temp
        Q_s=state
        olds=s
        olda=a
        ct=0

        while 1:

            #print(olds,s)
            state_diff=0
            if a==0:
                s=s
                temp=temp
            elif a==1:
                s+=1
                temp+=1.425
            elif a==2:
                s-=1
                temp-=1.425
            print(state)
            for j in range(len(state)):
                if j==0:
                    ss=get_state(bins,input4[i+1])
                else:
                    ss=state[j]
                #print(ss,state)
                state_diff+=ss*(16**j)
            #print(state_diff)
            state_diff=s-state_diff
            gauss1=gauss(state_diff)

            if s<0:
                reward=-10
                update_q(s,olds,a,olda,reward)
                break
            if s>=65536:
                reward=-10
                update_q(s,olds,a,olda,reward)
                break
                #reward=-10


            if ct > 500: #penalty for time over (RbLA)
                reward = -10
                update_q(s,olds,a,olda,reward)
                break
            if state_diff==0: #reward for acheiving goal
                reward = 10
                print(s,olds)
                update_q(s,olds,a,olda, reward) #(RbLA)
                break
            elif a!=0:#penalty for hit the wall #(RbLA: Optional)
                reward =gauss1-0.5
            elif a==0:
                reward=gauss1-1
            olda = a #storing previous action
            a = selectAction(s) #Selecting action
            update_q(s,olds,a,olda, reward) #update Q-value
            olds = s #storing previous action
            ct += 1 #step counter

print(Qtable[43706])

#結果
temp=(get_state(bins,input4[0])*1.425)-16.0
for i in range(100):
    #現在の周囲の気温を離散化して取得
    state=[]
    state.append(get_state(bins,input4[i]))
    state.append(get_state(bins,input1[i]))
    state.append(get_state(bins,input2[i]))
    state.append(get_state(bins,input3[i]))
    state_diff=0
    state_index=0
    for j in range(len(state)):
        state_index+=state[j]*(16**j)
    s=state_index
    a=selectAction(s)
    #temp=temp
    Q_s=state
    olds=s
    olda=a
    ct=0
    temp=(get_state(bins,input4[0])*1.425)-16.0
    for k in range(5):
        #print
        state_diff=0
        if(k==0):

            s=state_index
        action=selectAction(s)


        if action==0:
            s=s
            #temp=temp
        elif action==1:
            s+=1
            #temp+=1.425
        elif action==2:
            s-=1
            #temp-=1.425
        for j in range(len(state)):
            if j==0:
                ss=get_state(bins,input4[i+1])
            else:
                ss=state[j]
            state_diff+=ss*(16**j)

    print(state_diff,s)













