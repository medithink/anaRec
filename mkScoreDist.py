import math
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import comb

# ndarray配列の表示方法
# 有効数字5ケタ、表示桁数固定、指数表記の禁止
np.set_printoptions(precision=5, floatmode='fixed', suppress="True")

# 度数分布の刻む数の定義
# 10%ごとであれば10、5%ごとであれば20とc_numに定義する
c_num = 10
num = 100 / c_num   # 便宜上用いる。c_numを定義すればいじる必要はない。


# 変数の定義
per_cor = []
per_cor_f = []

# 正答率データの定義
# 2018年愛知卒試1正答率
per_cor = [ 958, 195, 915, 441, 720, 271, 34, 1000, 771, 729, 517, 695, 966, 559, 932, 712, 432, 669, 898, 627, \
            621, 641, 636, 695, 839, 585, 780, 915, 644, 771, 958, 559, 924, 941, 568, 576, 712, 559, 780, 390, \
            424, 585, 881, 195, 678, 186, 881, 559, 898, 34, 59, 881, 949, 822, 458, 780, 568, 847, 746, 568, 314, \
            534, 602, 703, 737, 907, 958, 898, 220, 771, 975, 331, 610, 585, 678, 983, 831, 619, 169, 669, 720, 720, \
            992, 881, 975, 424, 831, 966, 415, 712, 746, 915, 797, 1000, 661, 297, 339, 322, 390, 780, 975, 669, 466, \
            983, 949, 525, 119, 466, 831, 780, 568, 898, 305, 763, 661, 890, 610, 500, 610, 551, 466, 551, 169, 347, \
            822, 339, 831, 449, 297, 966, 288, 432, 873, 983, 763, 263, 364, 703, 941, 551, 797, 975, 958, 873, 542, \
            898, 941, 746, 492, 966, 305, 839, 881, 763, 797, 669, 619, 483, 949, 839, 432, 373, 814, 831, 475, 119, \
            966, 381, 415, 686, 814, 246, 958, 958, 678, 822, 458, 475, 644, 1000, 449, 864, 788, 729, 881, 881, 525, \
            763, 678, 864, 746, 669, 314, 797, 534, 941, 703, 441, 992, 958, 542, 992, 712, 949, 839, 339, 915, 169, \
            814, 720, 907, 356, 746, 593, 483, 814, 907, 398, 415, 754, 1000, 924, 627, 949, 551, 534, 992, 398, 737, \
            559, 822, 1000, 890, 212, 898, 458, 932, 924, 992, 1000]

# 度数分布を作成する
for i in range(len(per_cor)):
    per_cor_f.append(math.floor(per_cor[i] / (num * 10)) * num)

# 正答率度数分布のヒストグラムを作成
fig = plt.figure()
plt.hist(per_cor_f, bins = c_num + 1)
fig.show()

# 各要素の要素数をカウント
c = collections.Counter(per_cor_f)


# STEP1
# 各正答率ごとの点数確率分布を計算する
def calc_cor_dist(n, p):
    n = n # 正答率区分ごとの問題数。collections.counterにて取得
    p = p # 平均化した正答率。0~10%なら5%、0~5%なら2.5%とする(2019/01/14時点)
    temp_mat = np.zeros(n+1)
    for i in range(n+1):
        temp_mat[i] = comb(n, i, exact = True) * (p ** i) * ((1 - p) ** (n - i))
        # 正答率pの問題がn問あるとき、r問正解する確率は
        # n_C_r * p^r * (1-p)^(n-r)
        #  = comb(n, r, exact = True) * (p ** r) * ((1 - p) ** (n - r))
    
    temp_mat = temp_mat.reshape(1, n+1)   # 転置できるように行数、列数を明示
    return temp_mat

mat00 = calc_cor_dist(n = c[0], p = 0.05)
mat10 = calc_cor_dist(n = c[10], p = 0.15)
mat20 = calc_cor_dist(n = c[20], p = 0.25)
mat30 = calc_cor_dist(n = c[30], p = 0.35)
mat40 = calc_cor_dist(n = c[40], p = 0.45)
mat50 = calc_cor_dist(n = c[50], p = 0.55)
mat60 = calc_cor_dist(n = c[60], p = 0.65)
mat70 = calc_cor_dist(n = c[70], p = 0.75)
mat80 = calc_cor_dist(n = c[80], p = 0.85)
mat90 = calc_cor_dist(n = c[90], p = 0.95)
mat100 = calc_cor_dist(n = c[100], p = 0.9999)

# STEP2
# 各問題正答率ごとの行列を掛け合わせた点数分布を新たに計算する。
def calc_cor_comb(mat1, mat2):
    mat1 = mat1
    mat2 = mat2
    temp_mat = mat1.T * mat2    # 点数分布の計算結果
    res = np.zeros(len(mat1.T) + len(mat2.T) -1)    # 点数分布結果を格納するための配列
    for i in range(len(mat1.T) + len(mat2.T) -1):
        s = 0
        for j in range(i+1):
            k = i - j
            if j > len(temp_mat) -1 or k > len(temp_mat.T) -1 :
                continue
            s = s + temp_mat[j][k]
        res[i] = s
    res = res.reshape(1, len(res))
    return res

mat0010 = calc_cor_comb(mat00, mat10)
mat0020 = calc_cor_comb(mat0010, mat20)
mat0030 = calc_cor_comb(mat0020, mat30)
mat0040 = calc_cor_comb(mat0030, mat40)
mat0050 = calc_cor_comb(mat0040, mat50)
mat0060 = calc_cor_comb(mat0050, mat60)
mat0070 = calc_cor_comb(mat0060, mat70)
mat0080 = calc_cor_comb(mat0070, mat80)
mat0090 = calc_cor_comb(mat0080, mat90)
mat0000 = calc_cor_comb(mat0090, mat100)

# STEP3
# グラフに描画
fig = plt.figure()
x = list(range(len(mat0000.T)))
y = mat0000.tolist()

l95 = 0
s = 0
for i in range(len(y[0])):
    s = s + y[0][i]
    if s > 0.05:
        l95 = i
        break

plt.plot(x, y[0])
plt.plot(l95, y[0][l95])
fig.show()

l95



# STEP 4
# 点数を正答率にて傾斜した際の点数分布を計算する

# 点数対応表を作成
s_mat00 = []
s_mat00 = np.zeros((len(mat00.T)))
for i in range(len(mat00.T)):
    s_mat00[i] = i * 0.05

s_mat00 = s_mat00.reshape(len(mat00.T), 1)
s_mat00

s_mat10 = []
s_mat10 = np.zeros(len(mat10.T))
for i in range(len(mat10.T)):
    s_mat10[i] = i * 0.15

s_mat10 = s_mat10.reshape(1, len(mat10.T))
s_mat10


temp = s_mat00 + s_mat10
np.where(0.15, True, 0)
temp * np.where(temp <= 0.151, True, 0)

