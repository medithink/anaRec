import math
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import comb
# 自作moduleのimport
from mk import mk_path


# ndarray配列の表示方法
# 有効数字5ケタ、表示桁数固定、指数表記の禁止
np.set_printoptions(precision=5, floatmode='fixed', suppress="True")

# 正答率データの定義
# 正答率データの読み込み
# 112回必修一般
perCor112HI = []
filename = mk_path("112HI.txt") # 読み込むファイルの相対pathを指定
f = open(filename, encoding="utf-8-sig").read()
perCor112HI = f.split("\n")
# 112回必修臨床
perCor112HR = []
filename = mk_path("112HR.txt") # 読み込むファイルの相対pathを指定
f = open(filename, encoding="utf-8-sig").read()
perCor112HR = f.split("\n")
# 112回総論各論
perCor112SK = []
filename = mk_path("112SK.txt") # 読み込むファイルの相対pathを指定
f = open(filename, encoding="utf-8-sig").read()
perCor112SK = f.split("\n")


# 各問題正答率ごとの行列を掛け合わせた点数分布を新たに計算する。
def calcCorComb(mat1, corP):  # 2つの行列を入力として受け取る
    mat1 = mat1                 # 1～n問目までの正答率分布（0点の確率がp_0、1点の確率がp_1...） => [p_0 p_1 p_2 ... p_n]
    mat2 = np.zeros((1, 2))     # 正答率分布を格納するmat2の定義
    mat2[0][0] = 1 - corP  # 正答率pの問題の正答率分布（0点の確率が1-p_(n+1)、1点の確率がp_(n+1)） => [1-p_(n+1) p_(n+1)]
    mat2[0][1] = corP
    temp_mat = mat1.T * mat2    # 点数分布の計算結果
        # (p_0)*(1-p_(n+1)) (p_0)*(p_(n+1)) => n問目までで0点 ∧ n+1問目で0点  n問目までで0点 ∧ n+1問目で1点
        # (p_1)*(1-p_(n+1)) (p_1)*(p_(n+1)) => n問目までで1点 ∧ n+1問目で0点  n問目までで1点 ∧ n+1問目で1点
        # ...
        # (p_n)*(1-p_(n+1)) (p_n)*(p_(n+1)) => n問目まででn点 ∧ n+1問目で0点  n問目まででn点 ∧ n+1問目で1点
    res = np.zeros(len(mat1.T) + 1)    # 点数分布結果を格納するための配列
    print(len(res) - 1)
    for i in range(len(res)):
        if i == 0:
            res[i] = temp_mat[0][0]
        elif i == len(res) - 1:
            res[i] = temp_mat[i-1][1]
        else:
            res[i] = temp_mat[i][0] + temp_mat[i-1][1]
    res = res.reshape(1, len(res))
    return res

def calcGetScoreDist(corList):
    temp_list = corList
    scoreDist = np.zeros((1, 2))
    for i in range(len(temp_list)):
        if i == 0:
            scoreDist[0][0] = 1 - int(temp_list[0]) / 1000
            scoreDist[0][1] = int(temp_list[0]) / 1000
        else:
            scoreDist = calcCorComb(scoreDist, int(temp_list[i]) / 1000)
    return scoreDist

scoreDist112HI = calcGetScoreDist(perCor112HI) 
scoreDist112HR = calcGetScoreDist(perCor112HR)
scoreDist112SK = calcGetScoreDist(perCor112SK) 


# 必修の一般臨床の点数傾斜を含めた点数分布を新たに計算する。
def mkScoreDistHR2Range200(scoreDist):
    temp_res = np.zeros(201)
    for i in range(151):
        if i % 3 == 0:
            temp_res[i] = scoreDist[0][int(i / 3)]
    return temp_res

def aaa(scoreDistHR, corP):
    mat2 = np.zeros((1, 2)) 
    mat2[0][0] = 1 - corP
    mat2[0][1] = corP
    scoreDistHR = scoreDistHR.reshape(1, 201)
    temp_mat = scoreDistHR.T * mat2
    res = np.zeros(201)
    for i in range(201):
        if i == 0:
            res[i] = temp_mat[i][0]
        elif i == 200:
            res[i] = temp_mat[i-1][1]
        else:
            res[i] = temp_mat[i][0] + temp_mat[i-1][1]
    res = res.reshape(1, len(res))
    print(res.sum())
    return res

wScoreDist112HR = mkScoreDistHR2Range200(scoreDist112HR)
print(wScoreDist112HR)


for i in range(50):
    wScoreDist112HR = aaa(wScoreDist112HR, int(perCor112HI[i]) / 1000)

# cutoff点数の計算式
def colCutoff(scoreDistList, cutoff):
    c = cutoff          # cutoffとする数値
    l = []
    l = scoreDistList   # 計算に用いる点数分布のリスト
    s = 0
    res = 0
    for i in range(len(l)):
        s = s + l[i]
        if s > c:
            res = i
            return res

# グラフに描画
def mk_plot(scoreDist):
    fig = plt.figure()
    x = list(range(len(scoreDist.T)))
    y = scoreDist.tolist()
    # cutoff点数を計算
    cutoff950 = 0   # 95%ライン
    cutoff925 = 0   # 92.5%ライン
    cutoff900 = 0   # 90%ライン
    cutoff875 = 0   # 87.5%ライン
    cutoff850 = 0   # 85%ライン
    cutoff950 = colCutoff(y[0], 0.050)   # 95%ライン
    cutoff925 = colCutoff(y[0], 0.075)   # 92.5%ライン
    cutoff900 = colCutoff(y[0], 0.100)   # 90%ライン
    cutoff875 = colCutoff(y[0], 0.125)   # 87.5%ライン
    cutoff850 = colCutoff(y[0], 0.150)   # 85%ライン
    print(cutoff950)
    print(cutoff925)
    print(cutoff900)
    print(cutoff875)
    print(cutoff850)
    # 平均点の計算
    scoreList = np.arange(len(scoreDist[0]))    # 0～満点までの連続した点数配列を作成
    scoreAve = np.average(scoreList, axis=0, weights=y[0]) # 点数分布に従った平均点の算出
    plt.plot(x, y[0])
    plt.plot([cutoff950, cutoff950], [0, scoreDist.max()], "k-")
    plt.plot([scoreAve, scoreAve], [0, scoreDist.max()], "k-")
    fig.show()

mk_plot(scoreDist112SK)

mk_plot(wScoreDist112HR)
