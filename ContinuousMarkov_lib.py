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
    
    def getQ(self, mu):#最適化用
        q = self.p.copy()
        #推移率行列を求める
        #(1)サービス率と推移確率との積をとる
        for i in range(len(self.q)):
            q[i] *= mu[i]
        #(2)対角要素に行和のマイナス値を入れる
        for i in range(len(q)):
            q[i][i] = np.sum(q[i]) * (-1)
        return q

    def getStationary_solve(self):#numpy.solveを使う場合
        #定常分布を求める
        q1 = self.q.copy()
        #(3)最終列に1を代入
        right = [0 for i in range(len(self.q))]
        right[-1] = 1 #最後の要素のみ1にする
        q1[:,-1] = 1 #最終列を1にする
        #(4)連立方程式を解く πP=0 => P^tπ=0
        self.pi = solve(q1.T, right)
        return self.pi
    
    def getStationary_opt(self, q):#最適化用
        #定常分布を求める
        q1 = q.copy()
        #(3)最終列に1を代入
        right = [0 for i in range(len(q))]
        right[-1] = 1 #最後の要素のみ1にする
        q1[:,-1] = 1 #最終列を1にする
        #(4)連立方程式を解く πP=0 => P^tπ=0
        pi = solve(q1.T, right)
        return pi
    
    def getStationary_inv(self):#逆行列を使う場合(Durret P165参照)
        #定常分布を求める
        q1 = self.q.copy()
        #(3)最終列に1を代入
        right = [0 for i in range(len(self.q))]
        right[-1] = 1 #最後の要素のみ1にする
        q1[:,-1] = 1 #最終列を1にする
        #(4)逆行列を使って定常分布を算出
        inv_q1 = np.linalg.pinv(q1)
        self.pi = np.dot(right, inv_q1)
        return self.pi
    
    def getOptimize(self, mu):
        q = self.getQ(mu)
        pi = self.getStationary_opt(q)
        opt_value = np.var(pi)
        return opt_value
    
    def getConstraint(self,mu):#平均時間の合計に近くなるようにeq条件をつける
        nearTs = 0.8#条件値
        Ts = [1/i for i in mu]#平均時間
        return nearTs - np.sum(Ts)
        