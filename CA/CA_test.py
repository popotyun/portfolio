import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

def get_duplicate_list_order(seq):
    seen = []
    return [x for x in seq if seq.count(x) > 1 and not seen.append(x) and seen.count(x) == 1]

def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]

def has_duplicates2(seq):
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    return len(seq) != len(unique_list)

def transposed_matrix(fin,fout):#csv転置関数
    df=pd.read_csv(fin)
    df=df.T
    df.to_csv(fout,header=False, index=False)

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

"""
out_data = "/Users/shimizutakumi/Documents/PythonWorks-4/mesh_data/test_output.csv"
in_data = "/Users/shimizutakumi/Documents/PythonWorks-4/mesh_data/test_input.csv"

f1=open(out_data,"r")
csv_read=csv.reader(f1)

f2=open(in_data,"r")
csv_read=csv.reader(f2)


trans_out_data="/Users/shimizutakumi/Documents/PythonWorks-4/mesh_data/test_output_transposed2.csv"
taras_in_data="/Users/shimizutakumi/Documents/PythonWorks-4/mesh_data/test_input_transposed2.csv"
f3=open(trans_out_data,"w")
csv_read=csv.writer(f3)
f4=open(taras_in_data,"w")
csv_read=csv.reader(f4)

transposed_matrix(out_data,trans_out_data)
transposed_matrix(in_data,taras_in_data)
"""
#------訓練データと教師データの取得--------

input_file1="/area1.csv"
#transposed_file="/Users/shimizutakumi/Documents/PythonWorks-4/mesh_data/test_output_transposed.csv"
f1=open(input_file1,"r")
file1 =csv.reader(f1)#入力データ
input_file2 = "area1.csv"
f2=open(input_file2,"r")
file2=csv.reader(f2)#教師データ
#transposed_matrix(input_file2,transposed_file)
#transposed_matrix(input_file2,input_file2)
x=[] #入力データ
y=[] #教師データ
cnt=0
for row in file1:#入力データ
    #print(row[0],type(row[0]))
    if(row[0]=="2018-12-31 00:00:00"):

        break
    else:
        x.append(row)

for row in file2:#教師データ
    if(row[0]=="2018-12-31 00:00:00"):
        break
    else:
        y.append(row)
#print(x[0],y[0])
#print(len(x[0]))
print(float(y[25][0]))

index=x[0].index('2018-12-31 23:00:00')
print(index)
x_train=[]
y_train=[]
array_x=[]
array_y=[]

for i in range(len(x)):
    array_x=[]
    array_y=[]
    for j in range(len(x[i])):

        #print(len(x[i]))
        if j==index:
            break
        else:
            array_x.append(x[i][j])
            array_y.append(y[i][j])

    x_train.append(array_x)
    y_train.append(array_y)
#print(len(x),len(x[0]))

#print(x_train[1],y_train[0],len(x_train[0]),len(y_train[0]))



#-------------データ取得----------------

"""
x_train:測定可能地点のみのデータ
[[1][0][0][0][0]
 [0][0][0][0][0]
 [0][0][0][0][0]
 [0][0][0][0][0]
 [0][0][0][0][1]]

1:測定可能 0.005*実測値+0.75の値
0:測定不可能 0.0を出力


"""
#初期化
x_train=np.array(x_train)
y_train=np.array(y_train)


print(float(y_train[25][0]))
cells=np.full((5,5),-1,dtype='float128')

next_cells=np.full((5,5),-1,dtype='float128')

print(cells,next_cells)
#CA---近傍のセルと自身のセルの中央値を求める

#近接のインデックスをまとめる
index=[]
for i in range(len(cells)):
    for j in range(len(cells[i])):

        index.append(moore_neighborhood(j,i))

print(len(index))

#処理順の配列の作成

measurable_cells=[[0,0],[4,4]]
predict_cells=[[0,0],[4,4]]
for k in range(len(cells)-1):
    for i in range(len(predict_cells)):
        nei = moore_neighborhood(predict_cells[i][1],predict_cells[i][0])
        for j in nei:
            predict_cells.append(j)

predict_cells=get_duplicate_list_order(predict_cells)
print(predict_cells,len(predict_cells))

#[[0, 0], [4, 4], [1, 0], [0, 1], [1, 1], [3, 4], [4, 3], [3, 3], [2, 0], [2, 1], [0, 2], [1, 2], [2, 2], [2, 4], [2, 3], [4, 2], [3, 2], [3, 0], [3, 1], [0, 3], [1, 3], [1, 4], [4, 1], [4, 0], [0, 4]] 25





"""
for i in range(len(x_train[1])):
    print(x_train[1][i])

"""
"""
cells[0][0]=x[1][0]
cells[4][4]=x[25][0]

print(cells)

"""


"""
t:現実時間
cells:kの状態
next_cells:k+1の状態
測定可能地点の状態は変えない
25地点の表示
正規化した値を気温に変換してグラフにプロット


"""
#print(y_train[5][0])

cnt=0
#plot=np.zeros((25,200))#プロット用
plot=[]
plot2=[]
print(float(y[25][0]))
for t in range(1):#現実時間　1時間ごと
    for i in measurable_cells:#測定可能セルの初期状態を決定 
        print("a",y_train[i[1]+1][t])
        #print(float(y[25][0]))
        mes_index=(5*i[1]+i[0])+1
        min_data=-30.0
        data=(float(y_train[mes_index][t])- min_data)/80
        print(data)
        cells[i[1]][i[0]]=data
        print(cells,type(cells[0][0]))

    for k in range(200):#内部時間<-ここを繰り返して安定してるかを確認
        plot=[]
        for o in range(len(cells)):
            for p in range(len(cells[o])):

                plot.append(cells[o][p])

        plot2.append(plot)

        for j in predict_cells:#推定順
            index=moore_neighborhood(j[1],j[0])#近傍セルの座標取得

            next_state=[]
            for l in index:#近傍セルのkの状態を取得
                next_state.append(cells[l[0]][l[1]])

            next_cells[j[0]][j[1]]=(max(next_state)+min(next_state))/2#k+1の状態更新の計算式

        for m in range(len(next_cells)):
            for n in range(len(next_cells[m])):
                if(m==0 and n==0) or (m==4 and n==4):
                    a=1
                else:
                    #print(m,n)
                    cells[m][n] = next_cells[m][n]

plot=[]
for o in range(len(cells)):
    for p in range(len(cells[o])):
        plot.append(cells[o][p])
print((0.37*80)-30)
plot2.append(plot)
plot2=np.array(plot2)
#plot2=plot2.T

print(cells)
plot=np.array(plot2)
plot=plot.T
print(plot,len(plot[0]),len(plot))

for i in range(len(plot)):
    for j in range(len(plot[i])):

        plot[i][j]=(80.0*plot[i][j])-30.0

for i in range(len(plot)):

    if(i==0 or i==24):
        a=0

    else:

        plot[i][0]=-1


print(len(plot),plot[4])
ax=[]
Figure=plt.figure()
for i in range(25):
    ax.append(Figure.add_subplot(5,5,i+1))

for i in range(25):
    ax[i].plot(plot[i])


plt.show()
"""
plt.plot(plot)
plt.show()

"""




















