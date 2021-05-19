import numpy as np
from numpy.linalg import solve

class ContinuousMarkov_lib:

    def __init__(self, p, mu):
        self.p = p
        self.mu = mu

    def getTransitionRate(self):
        self.q = self.p.copy()
        #推移率行列を求める
        #(1)サービス率と推移確率との積をとる
        for i in range(len(self.q)):
            self.q[i] *= self.mu[i]
        #(2)対角要素に行和のマイナス値を入れる
        for i in range(len(self.q)):
            self.q[i][i] = np.sum(self.q[i]) * (-1)
        return self.q

    def getStationary(self):
        #定常分布を求める
        q1 = self.q.copy()
        #(3)最終列に1を代入
        right = [0 for i in range(len(self.q))]
        right[-1] = 1 #最後の要素のみ1にする
        q1[:,-1] = 1 #最終列を1にする
        #(4)連立方程式を解く πP=0 => P^tπ=0
        self.pi = solve(q1.T, right)
        return self.pi