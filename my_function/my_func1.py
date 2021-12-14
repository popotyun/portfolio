import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import itertools

def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]



def moore_neighborhood(x,y):
    #ムーア関数　被りあり．returnされたインデックスから重複削除する必要あり
    #print(y,x)
    index=[]
    upper=[y-1,x]
    under=[y+1,x]
    left=[y,x-1]
    right=[y,x+1]
    right_upper=[y-1,x+1]
    right_under=[y+1,x+1]
    left_upper=[y-1,x-1]
    left_under=[y+1,x-1]
    #index= [y,x]
    """
    if is_out_of_range(arr,upper):
        
        index.append([upper,x])

    if is_out_of_range(arr,under):
        index.append([upper,x])

    if is_out_of_range(arr,left):
        index.append([y,left])

    if is_out_of_range(arr,right):
        index.append([y,right])

    if is_out_of_range(arr,right_upper):
        index.append([right_upper])

    if is_out_of_range(arr,right_under):
        index.append([right_under])

    if is_out_of_range(arr,left_upper):
        index.append([left_upper])

    if is_out_of_range(arr,left_under):
        index.append([left_under])

    return index

"""
    #print(upper,under,left,right,left_upper,left_under,right_upper,right_under)
    index.append([y,x])

    if upper[0] > -1:
        index.append(upper)
    if under[0] < 5:
        index.append(under)

    if left[1] > -1:
        index.append(left)

    if right[1] < 5:
        index.append(right)

    if left_upper[0] > -1 and left_upper[1] > -1:
        index.append(left_upper)

    if left_under[0] < 5 and left_under[1] > -1:
        index.append(left_under)

    if right_upper[0] > -1 and right_upper[1] < 5:
        index.append(right_upper)

    if right_under[0] < 5 and right_under[1] < 5:
        index.append(right_under)


    return index

def get_duplicate_list_order(seq):
    seen = []
    return [x for x in seq if seq.count(x) > 1 and not seen.append(x) and seen.count(x) == 1]


def sigmoid (x):
    return 1/(1+np.exp(-x))

def dataload(d,l_in):
    x=np.zeros(3)
    for i in range(3):
        x[i]=l_in[d][i]
    return x
    

#入力層用の関数
def forward(x,w,b,mid):
    s=np.zeros(mid)
    h=np.zeros(mid)
    #print(x,w,b,mid)
    #print(w)
    for j in range(mid):
        for i in range(len(x)):

            s[j]+=w[i][j]*x[i]
        s[j]+=b[j]
        h[j]=sigmoid(s[j])
    return h

#出力層用の関数
def forward_2(h,w,b,mid,t):
    e=0
    o=np.zeros((1))
    y=np.zeros((1))
    for k in range(1):
        for j in range(mid):
            o[k]+=w[j][k]*h[j] 
        o[k]+=b[k]
        y[k]=sigmoid(o[k])
        e+=(y[k]-t)**2
    e=1/2*e     
    return y,e

#backを求める関数
def back(eta,y,t,h,w2,mid):
    back2=np.zeros(mid)
    back1=np.zeros(1)
    for k in range(1):
        back1[k]=(y[k]-t)*(1.0-y[k])*y[k]

    for k in range(1):
        for j in range(mid):
            back2[j]+=back1[k]*h[j]*w2[j][k]

    return back1,back2

#逆伝搬
def backforward(b2,w2,b1,w1,eta,back1,back2,mid,h,x):

    for k in range(1):
        b2[k]=b2[k]-eta*back1[k]
            
    for k in range(1):            
        for j in range(mid):
            w2[j][k]=w2[j][k]-eta*back1[k]*h[j] 
            

    for j in range(mid):                
        b1[j]=b1[j]-eta*(1.0-h[j])*h[j]*back2[j]

    for j in range(mid):               
        for i in range(len(w1)):
            w1[i][j]=w1[i][j]-eta*(1.0-h[j])*h[j]*back2[j]*x[i]
            
    return b2,w2,b1,w1


def dataload(file,date):
    #file:取得したいデータ
    #date:どこまでのデータか
    f1=open(file,"r")
    file1 =csv.reader(f1)#入力データ


    x=[] #入力データ
    cnt=0
    for row in file1:#入力データ
        #print(row[0],type(row[0]))
        if(row[0]==date):

            break
        else:
            x.append(row)

    index=x[0].index(date)
    print(index)
    x_train=[]
    y_train=[]
    array_x=[]
    array_y=[]

    for i in range(len(x)):
        array_x=[]
        for j in range(len(x[i])):

            #print(len(x[i]))
            if j==index:
                break
            else:
                array_x.append(x[i][j])

        x_train.append(array_x)

    return x_train



