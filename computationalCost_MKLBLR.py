import os
import numpy as np
os.sys.path.append('lib')
from lib.RFF_BLR import RFF_BLR
import time
from sklearn.datasets import make_regression
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state

N_array = np.arange(50,1001,50)
C_array = np.arange(2,17)
M_perc = [.1,  .2 ,.3, .4, .5, .6, .7, .8, .9, 1, 
                    1.1, 1.2, 1.5, 2, 3,5]
D = 50
n_ini = 1
rep = 5

def make_poly_regression(n_samples=100,
    n_features=100,
    *,
    n_informative=20,
    n_targets=1,
    bias=0.0,
    noise=0.0,
    random_state=None,
    interaction_only=True,
    degree=2,
    include_bias=False):

    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)

    X = generator.standard_normal(size=(n_samples, n_features))

    poly = PolynomialFeatures(interaction_only=interaction_only,
                              degree=degree,
                              include_bias=include_bias)
    Xp = poly.fit_transform(X)
                          
    W = np.zeros((Xp.shape[1], n_targets))
    W[:n_informative, :] = generator.standard_normal(size=(n_informative, n_targets))
    W = W[:, generator.permutation(W.shape[1])]

    y = np.dot(Xp, W) + bias

    # Add noise
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)
    
    y = np.squeeze(y)

    return X, y

comp_dict = {}
# MKL-BLR
print('MKL-BLR')

filename = 'Results/computational_cost_MKLBLR_D'+str(D)+'.pkl'
if os.path.exists(filename):
    print ("Loading existing model...")
    dict = pickle.load(open(filename, "rb" ) )
else:
    dict ={'time': np.zeros((len(N_array),len(C_array),len(M_perc), rep)),
            'R2': np.zeros((len(N_array),len(C_array),len(M_perc), rep)),
            'components': np.zeros((len(N_array),len(C_array),len(M_perc), rep)),
            'components_lin': np.zeros((len(N_array),len(C_array),len(M_perc), rep)),
            'components_rff': np.zeros((len(N_array),len(C_array),len(M_perc), rep)),
            }

model = RFF_BLR()

for n,N_tr in enumerate(N_array):
    N_tst = int(N_tr*(0.4/0.6)) #60% train, 40% test
    for c,C in enumerate(C_array):
        X, Y = make_poly_regression(N_tr+N_tst, D, 
                n_targets=C,
                random_state=42,
                interaction_only=True,
                degree=4,
                include_bias=False,
                noise = 2.)
        X_tr, Y_tr = X[:N_tr,:], Y[:N_tr,:]
        X_tst, Y_tst = X[N_tr:,:], Y[N_tr:,:]
        for m,M in enumerate(M_perc):
            Kc = int(N_tr*M)
            print('Computing for %d training samples, %d tasks and %d components' %(N_tr, C, Kc))
            for r in np.arange(rep):
                if dict['R2'][n,c,m,r] == 0:
                    tic = time.time()
                    # for i in np.arange(n_iter):
                    model.fit(X_tr, Y_tr, X_tst, Y_tst, n_components = Kc, maxit = 1000, links = True, gamma_param = True)
                    dict['time'][n,c,m,r] = time.time()-tic
                    dict['R2'][n,c,m,r] = model.R2_tst[-1]
                    dict['components'][n,c,m,r] = model.q_dist.K
                    dict['components_lin'][n,c,m,r] = model.input_idx[Kc:].sum()
                    dict['components_rff'][n,c,m,r] = model.input_idx[:Kc].sum()
                    print('Time: %.2f, R2: %.3f, K: %d, K lin: %d, K RFF: %d' %(np.mean(dict['time'][n,c,m,:]), 
                                                                                np.mean(dict['R2'][n,c,m,:]),
                                                                                np.mean(dict['components'][n,c,m,:]),
                                                                                np.mean(dict['components_lin'][n,c,m,:]),
                                                                                np.mean(dict['components_rff'][n,c,m,:])))            
            with open(filename, 'wb') as handle:
                pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)