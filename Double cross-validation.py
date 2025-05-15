# -*- coding: utf-8 -*-
"""
Created on Thu May 15 22:29:03 2025

@author: horik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import math
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.simplefilter('ignore')
from numpy import matlib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn import tree, svm, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel,Matern,DotProduct

#PLS Parameter
max_number_of_components = 20
#RR Parameter
rr_lambdas = 2 ** np.arange(-5, 9, dtype = float)
#LASSO Parameter
lasso_lambdas = 2 ** np.arange(-15, -1, dtype = float)
#ElsticNet Parameter
en_lambdas= 2** np.arange(-15, 15, 1, dtype = float)
en_alpha=np.arange(0.01, 1, 0.01)
#LSVR Parameter
linear_svr_cs=2**np.arange(-5, 5, 1.0)
linear_svr_epsilons=2**np.arange(-10, 1, 1.0)
#NSVR parameter
nonlinear_svr_cs=2**np.arange(-5, 10, 1.0)
nonlinear_svr_epsilons=2**np.arange(-10, 1, 1.0)
nonlinear_svr_gammas=2**np.arange(-20, 11, 1.0)
#DT Parameter
max_max_depth = 10  
min_samples_leaf = 2
#RF Parameter
ratios_of_x=np.arange(0.1, 1.1, 0.1)
rf_n_estimators=300
#GPR Parameter
alpha = 1e-10

#Regression model
regression_model_method = ['OLS','PLS','RR','LASSO','EN','LSVR','NSVR','DT',
                           'RF','GPR0','GPR1','GPR2','GPR3','GPR4','GPR5',
                           'GPR6','GPR7','GPR8','GPR9','GPR10','GBDT','XGB','LGB'
                           ]

#Data
df = pd.read_csv('logSdataset1290.csv')
x = df.iloc[:,2:]
y = df.iloc[:,1]

#Save r2,MAE,RMSE
DCV_values = np.zeros([len(regression_model_method), 3])
DCV_values = pd.DataFrame(DCV_values,index=regression_model_method,columns=['r2_DCV','RMSE_DCV','MAE_DCV'])

#Save predicted values
predicted_y_values = np.zeros([len(y), len(regression_model_method)])
predicted_y_values = pd.DataFrame(predicted_y_values, index=y.index, columns=regression_model_method)

inner_fold_number = 5
outer_fold_number = 5

#Split outer-cv
min_number = math.floor(x.shape[0] / outer_fold_number)
mod_number = x.shape[0] - min_number * outer_fold_number
index = np.matlib.repmat(np.arange(1, outer_fold_number+1, 1), 1, min_number).ravel()
if mod_number != 0:
    index = np.r_[index, np.arange(1, mod_number+1, 1)]
np.random.seed(999)
fold_index_in_outer_cv = np.random.permutation(index)

#Model construction
for regression_model_methods in regression_model_method:
    print(regression_model_methods)
    
    #Outer-cv
    estimated_y_in_outer_cv = np.zeros(len(y))
    for fold_number_in_outer_cv in np.arange(1, outer_fold_number+1, 1):
        print(fold_number_in_outer_cv, '/', outer_fold_number)
        
        #Split training and test data
        x_train_in_outer_cv = x.iloc[fold_index_in_outer_cv != fold_number_in_outer_cv, :]
        y_train = y[fold_index_in_outer_cv != fold_number_in_outer_cv]
        x_test_in_outer_cv = x.iloc[fold_index_in_outer_cv == fold_number_in_outer_cv, :]

        x_train = x_train_in_outer_cv.copy()
        x_test = x_test_in_outer_cv.copy()
        
        x_train = x_train.drop(x_train.columns[np.where(x_train.var()==0)],axis=1)
        x_test = x_test[x_train.columns]

        #Autoscaling
        autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
        autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
        autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
        
        autoscaled_x_train = autoscaled_x_train.values
        autoscaled_y_train = autoscaled_y_train.values
        autoscaled_x_test = autoscaled_x_test.values
        
        #OLS
        if regression_model_methods == 'OLS':
            regression_model = LinearRegression()
        
        #PLS
        elif regression_model_methods == 'PLS':
            pls_components = np.arange(1,min(np.linalg.matrix_rank(autoscaled_x_train)+1,max_number_of_components+1),1)
            r2all = list()
            r2cvall = list()
            for pls_component in pls_components:
                pls_model_in_cv = PLSRegression(n_components=pls_component)
                pls_model_in_cv.fit(autoscaled_x_train, autoscaled_y_train) 
                calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_x_train))
                estimated_y_in_cv = np.ndarray.flatten(
                    model_selection.cross_val_predict(pls_model_in_cv, autoscaled_x_train, autoscaled_y_train, cv=inner_fold_number))
                calculated_y_in_cv = calculated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
        
                r2all.append(float(1 - sum((y_train - calculated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
                r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))

            optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall)) 
            optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
            regression_model = PLSRegression(n_components=optimal_pls_component_number)
    
        #RR
        elif regression_model_methods == 'RR':
            r2cvall = list()
            for ridge_lambda in rr_lambdas:
                rr_model_in_cv = Ridge(alpha=ridge_lambda)
                estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                      cv=inner_fold_number) 
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
                r2cvall.append(float(1 - sum((y_train - np.ndarray.flatten(estimated_y_in_cv)) ** 2) / sum((y_train - y_train.mean()) ** 2)))
            optimal_ridge_lambda = rr_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]] 
            regression_model = Ridge(alpha=optimal_ridge_lambda)

        #LASSO
        elif regression_model_methods == 'LASSO':
            r2cvall = list()
            for lasso_lambda in lasso_lambdas:
                lasso_model_in_cv = Lasso(alpha=lasso_lambda) 
                estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                      cv=inner_fold_number)
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
                r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))) 
            optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]] 
            regression_model = Lasso(alpha=optimal_lasso_lambda) 
            
        #EN
        elif regression_model_methods == 'EN':
            elastic_net_in_cv = ElasticNetCV(cv=inner_fold_number, l1_ratio=en_alpha, alphas=en_lambdas)
            elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_y_train) 
            optimal_elastic_net_alpha = elastic_net_in_cv.alpha_ 
            optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_ 
            regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
            
        #LSVR
        elif regression_model_methods == 'LSVR':
            linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                            cv=inner_fold_number)
            linear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
            optimal_linear_svr_c = linear_svr_in_cv.best_params_['C'] 
            optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon'] 
            regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon) 
            
        #NSVR
        elif regression_model_methods == 'NSVR':
            variance_of_gram_matrix = list()
            numpy_autoscaled_x_train = np.array(autoscaled_x_train)
            for nonlinear_svr_gamma in nonlinear_svr_gammas:
                gram_matrix = np.exp(-nonlinear_svr_gamma * ((numpy_autoscaled_x_train[:, np.newaxis] - numpy_autoscaled_x_train) ** 2).sum(axis=2))
                variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
            
            optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
            nonlinear_svr_in_cv = GridSearchCV(svm.SVR(kernel='rbf', gamma=optimal_nonlinear_gamma),{'C': nonlinear_svr_cs, 'epsilon': nonlinear_svr_epsilons}, cv=inner_fold_number)
            nonlinear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
            optimal_nonlinear_c = nonlinear_svr_in_cv.best_params_['C']
            optimal_nonlinear_epsilon = nonlinear_svr_in_cv.best_params_['epsilon']
            print(optimal_nonlinear_gamma, optimal_nonlinear_epsilon, optimal_nonlinear_c)
            regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,gamma=optimal_nonlinear_gamma)               

        #DT
        elif regression_model_methods == 'DT':
            rmse_cv = []
            max_depthes = []
            for max_depth in range(1, max_max_depth):
                model_in_cv = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                estimated_y_in_cv = model_selection.cross_val_predict(model_in_cv,x_train,y_train,cv=inner_fold_number)
                rmse_cv.append((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)
                max_depthes.append(max_depth)
            optimal_max_depth = max_depthes[rmse_cv.index(max(rmse_cv))]
            regression_model = tree.DecisionTreeRegressor(max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf)

        # RF
        elif regression_model_methods == 'RF':
            rmse_oob_all = list()
            for random_forest_x_variables_rate in ratios_of_x:
                RandomForestResult = RandomForestRegressor(n_estimators=rf_n_estimators, max_features=int(
                    max(math.ceil(x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True,random_state=999)
                RandomForestResult.fit(autoscaled_x_train, autoscaled_y_train)
                estimated_y_in_cv = RandomForestResult.oob_prediction_
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean() 
                rmse_oob_all.append((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5) 
            optimal_random_forest_x_variables_rate = ratios_of_x[
                np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]] 
            regression_model = RandomForestRegressor(n_estimators=rf_n_estimators, max_features=int(
                max(math.ceil(x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True,random_state=999)             

        #GPR0
        elif regression_model_methods == 'GPR0':
            kernel = ConstantKernel() * DotProduct() + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GRP1
        elif regression_model_methods == 'GPR1':
            kernel = ConstantKernel() * RBF() + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR2
        elif regression_model_methods == 'GPR2':
            kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR3
        elif regression_model_methods == 'GPR3':
            kernel = ConstantKernel() * RBF(np.ones(x_train.shape[1])) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR4
        elif regression_model_methods == 'GPR4':
            kernel = ConstantKernel() * RBF(np.ones(x_train.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR5
        elif regression_model_methods == 'GPR5':
            kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR6
        elif regression_model_methods == 'GPR6':
            kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR7
        elif regression_model_methods == 'GPR7':
            kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR8
        elif regression_model_methods == 'GPR8':
            kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR9
        elif regression_model_methods == 'GPR9':
            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GPR10
        elif regression_model_methods == 'GPR10':
            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        #GBDT
        elif regression_model_methods == 'GBDT':
            regression_model = GradientBoostingRegressor(random_state=999)
            
        #XGB
        elif regression_model_methods == 'XGB':
            regression_model = xgb.XGBRegressor(random_state=999)
        
        #LGB
        elif regression_model_methods == 'LGB':
            regression_model = lgb.LGBMRegressor(random_state=999)

        regression_model.fit(autoscaled_x_train, autoscaled_y_train)
        
        #Split outer data
        predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_x_test))
        predicted_ytest = predicted_ytest * y_train.std(ddof=1) + y_train.mean()
        estimated_y_in_outer_cv[fold_index_in_outer_cv==fold_number_in_outer_cv] = predicted_ytest
    
    predicted_y_values[regression_model_methods] = estimated_y_in_outer_cv[:]
    
    #yyplot
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=figure.figaspect(1))
    plt.title(f'{regression_model_methods}')
    plt.scatter(y,estimated_y_in_outer_cv) 
    y_max = y.max()
    y_min = y.min()
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
              [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') 
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) 
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) 
    plt.xlabel('Actual Y') 
    plt.ylabel('Predicted Y in DCV') 
    plt.show()

    #Calculate evaluation score
    evaluation = np.zeros((3, 1))
    evaluation[0,0] = float(1 - sum((y - estimated_y_in_outer_cv) ** 2) / sum((y - y.mean()) ** 2)) #r2
    evaluation[1,0] = float((sum((y - estimated_y_in_outer_cv) ** 2) / len(y)) ** 0.5) #RMSE
    evaluation[2,0] = float(sum(abs(y - estimated_y_in_outer_cv)) / len(y)) #MAE
    DCV_values.loc[regression_model_methods,:] = evaluation[:,0] 

#Save results
predicted_y_values = pd.concat([y,predicted_y_values],axis=1)
predicted_y_values.to_csv('predicted_y_values.csv')
