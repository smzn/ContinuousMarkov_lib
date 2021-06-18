import numpy as np
import ContinuousMarkov_lib as mdl
import utils
import scipy.optimize

#Durrett P165 Example 4.14 (Duke Basketball)
mu = np.array([4.0, 15/2, 3.0, 10.0]) #サービス率
p = np.array([[0.0, 0.7, 0.3, 0.0],[0.0, 0.0, 1.0, 0.0],[1/3, 0.0, 0.0, 2/3], [1.0, 0.0, 0.0, 0.0]])

#WiFi201409モデル
'''
service_rate = utils.getCSV('./csv/ServiceRate.csv') #今回ヘッダーなしで取り込み
mu =service_rate.values #numpyに変換
tp = utils.getCSV('./csv/probability_201409.csv') #今回ヘッダーなしで取り込み
p = tp.values
'''

clib = mdl.ContinuousMarkov_lib(p, mu)
q = clib.getTransitionRate()
print('Transition Rate : {}'.format(q[0]))
pi_solve = clib.getStationary_solve()
print('Solveでの定常分布 : {}'.format(pi_solve))

pi_inv = clib.getStationary_inv()
print('逆行列での定常分布 : {}'.format(pi_inv))

#最適化
cons = ({'type':'eq','fun':clib.getConstraint})#条件
opt=scipy.optimize.minimize(clib.getOptimize,x0=mu,method='CG',constraints=cons)#,bounds=b1
print(opt['x'])
olib = mdl.ContinuousMarkov_lib(p, opt['x'])
q = olib.getTransitionRate()
pi_solve = olib.getStationary_solve()
print('最適化後の定常分布 : {}'.format(pi_solve))