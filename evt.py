# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:09:47 2025
@author: yangzhibo
"""

from scipy.optimize import minimize
from math import *
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

class SPOT:
    def __init__(self, q = 1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0
    
    def fit(self,init_data,data):
        if isinstance(data,list):
            self.data = np.array(data)
        elif isinstance(data,np.ndarray):
            self.data = data
        elif isinstance(data,pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return
            
        if isinstance(init_data,list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data,np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data,pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data,int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data,float) & (init_data<1) & (init_data>0):
            r = int(init_data*data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def initialize(self, level = 0.98, verbose = False):
        level = level-floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold 
        self.Nt = self.peaks.size
        self.n = n_init
        g,s,l = self._grimshaw()
        self.extreme_quantile = self._quantile(g,s)
        return 

    def _rootsFinder(fun,jac,bounds,npoints,method):
        if method == 'regular':
            if bounds[1] == bounds[0]:
                X0 = [float(bounds[0])]
            else:
                step = (bounds[1]-bounds[0])/(npoints+1)
                X0 = np.arange(bounds[0]+step,bounds[1],step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0],bounds[1],npoints)
        
        def objFun(X,f,jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g+fx**2
                j[i] = 2*fx*jac(x)
                i = i+1
            return g,j
        
        opt = minimize(lambda X:objFun(X,fun,jac), X0, method='L-BFGS-B', jac=True, bounds=[bounds]*len(X0))
        X = opt.x
        np.round(X,decimals = 5)
        return np.unique(X)
    
    def _log_likelihood(Y,gamma,sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma/sigma
            L = -n * log(sigma) - ( 1 + (1/gamma) ) * ( np.log(1+tau*Y) ).sum()
        else:
            L = n * ( 1 + log(Y.mean()) )
        return L

    def _grimshaw(self,epsilon = 1e-8, n_points = 10):
        def u(s):
            return 1 + np.log(s).mean()
            
        def v(s):
            return np.mean(1/s)
        
        def w(Y,t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            return us*vs-1
        
        def jac_w(Y,t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            jac_us = (1/t)*(1-vs)
            jac_vs = (1/t)*(-vs+np.mean(1/s**2))
            return us*jac_vs+vs*jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()
        a = -1/YM
        if abs(a)<2*epsilon:
            epsilon = abs(a)/n_points
        
        a = a + epsilon
        b = 2*(Ymean-Ym)/(Ymean*Ym)
        c = 2*(Ymean-Ym)/(Ym**2)
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks,t), lambda t: jac_w(self.peaks,t), (a+epsilon,-epsilon), n_points,'regular')
        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks,t), lambda t: jac_w(self.peaks,t), (b,c), n_points,'regular')
        zeros = np.concatenate((left_zeros,right_zeros))
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks,gamma_best,sigma_best)
        for z in zeros:
            gamma = u(1+z*self.peaks)-1
            sigma = gamma/z
            ll = SPOT._log_likelihood(self.peaks,gamma,sigma)
            if ll>ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll
        return gamma_best,sigma_best,ll_best

    def _quantile(self,gamma,sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma/gamma)*(pow(r,-gamma)-1)
        else:
            return self.init_threshold - sigma*log(r)

    def run_simp(self, with_alarm = True):
        th = []
        suoyin = np.array(range(self.data.shape[0]))
        dayu = suoyin[self.data > self.init_threshold]
        self.n = self.data.size
        self.peaks = np.append([], self.data[self.data > self.init_threshold] - self.init_threshold)
        self.Nt = self.data[self.data > self.init_threshold].shape[0]
        g,s,l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)
        th.append(self.extreme_quantile)
        return {'thresholds' : th, 'alarms': []}

##################################################################################

from scipy import stats
def auto_detect_threshold(loss_list_use):
    """
    自动检测loss分布的阈值，不依赖预设的q值
    """
    losses = np.array(loss_list_use)
    
    # 方法1: 基于分布峰度和偏度
    skewness = stats.skew(losses)
    kurtosis = stats.kurtosis(losses)
    
    # 方法2: 使用混合高斯模型拟合
    try:
        print('nice')
        from sklearn.mixture import GaussianMixture
        losses_reshaped = losses.reshape(-1, 1)
        
        # 尝试2-3个成分的GMM
        best_bic = np.inf
        best_gmm = None
        
        for n_components in [2, 3]:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(losses_reshaped)
            bic = gmm.bic(losses_reshaped)
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        
        # 找到最高均值的组分作为异常分布
        means = best_gmm.means_.flatten()
        stds = np.sqrt(best_gmm.covariances_.flatten())
        weights = best_gmm.weights_
        
        # 选择权重较小但均值较高的组分
        anomaly_component = np.argmax(means)
        threshold = means[anomaly_component] - 2 * stds[anomaly_component]
        
    except:
        # 方法3: 基于统计的fallback方法
        print('bad')
        Q1 = np.percentile(losses, 25)
        Q3 = np.percentile(losses, 75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
    
    return max(threshold, np.min(losses))



if __name__ == '__main__':
    # 生成测试数据 - 模拟网络流量数据
    np.random.seed(42)
    
    # 正常流量数据（大部分数据）
    normal_traffic = np.random.normal(100, 15, 1000)
    
    # 异常流量数据（少量异常点）
    anomalies = np.random.uniform(200, 300, 20)
    
    # 合并数据
    all_data = np.concatenate([normal_traffic, anomalies])
    np.random.shuffle(all_data)
    
    # 划分初始训练数据和测试数据
    init_data_size = 800
    init_data = all_data[:init_data_size]
    test_data = all_data[init_data_size:]
    
    print(f"总数据量: {len(all_data)}")
    print(f"初始训练数据量: {len(init_data)}")
    print(f"测试数据量: {len(test_data)}")
    print(f"异常点数量: {len(anomalies)}")
    
    
    # 创建SPOT实例
    spot = SPOT(q=1e-2)  # 设置异常概率为0.1%
    
    # 拟合数据
    spot.fit(init_data, test_data)
    
    # 初始化SPOT（计算初始阈值）
    spot.initialize(level=0.98)
    
    print(f"初始阈值: {spot.init_threshold:.2f}")
    print(f"极端分位数阈值: {spot.extreme_quantile:.2f}")
    
    # 运行检测
    result = spot.run_simp()
    threshold = result['thresholds'][0]
    
    threshold = auto_detect_threshold(init_data)
    
    
    print(f"最终异常检测阈值: {threshold:.2f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    # 绘制所有数据点
    plt.plot(range(len(all_data)), all_data, 'b.', alpha=0.6, label='data point')
    
    # 标记异常点
    anomalies_indices = np.where(all_data > threshold)[0]
    plt.plot(anomalies_indices, all_data[anomalies_indices], 'r.', alpha=0.8, label='Detected anomaly')
    
    # 绘制阈值线
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Abnormal Threshold ({threshold:.2f})')
    plt.axhline(y=spot.init_threshold, color='g', linestyle='--', label=f'Initial threshold ({spot.init_threshold:.2f})')
    
    # 标记训练数据和测试数据的分界线
    plt.axvline(x=init_data_size, color='k', linestyle='-', alpha=0.5, label='Training/Testing Boundary')
    
    plt.xlabel('Time Index')
    plt.ylabel('Flow value')
    plt.title('SPOT Anomaly Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 计算性能指标
    true_anomalies = np.where(all_data > np.percentile(normal_traffic, 99))[0]  # 真实异常（正常数据的99%分位数以上）
    detected_anomalies = anomalies_indices
    
    # 确保只考虑测试数据
    true_anomalies = true_anomalies[true_anomalies >= init_data_size]
    detected_anomalies = detected_anomalies[detected_anomalies >= init_data_size]
    
    # 计算精确率、召回率和F1分数
    tp = len(np.intersect1d(true_anomalies, detected_anomalies))  # 真阳性
    fp = len(np.setdiff1d(detected_anomalies, true_anomalies))    # 假阳性
    fn = len(np.setdiff1d(true_anomalies, detected_anomalies))    # 假阴性
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n性能评估:")
    print(f"真阳性 (TP): {tp}")
    print(f"假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn}")
    print(f"精确率: {precision:.3f}")
    print(f"召回率: {recall:.3f}")
    print(f"F1分数: {f1:.3f}")


