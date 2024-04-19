import os
import numpy as np
os.sys.path.append('lib')
import pickle

import torch
import torch.optim as optim
from torchmetrics.functional import r2_score as r2_torch
from sklearn.metrics import r2_score

from lib.RVFL import RVFL
from lib.RFF_BLR import RFF_BLR

def train_ELM_lg(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):

    # Transforming to torch tensors
    X_tr = torch.from_numpy(X_tr).float()
    X_tst = torch.from_numpy(X_tst).float()
    Y_tr = torch.from_numpy(Y_tr).float()
    Y_tst = torch.from_numpy(Y_tst).float()

    D = X_tr.shape[1]

    # Initialize RVFL model
    model = RVFL(X_tr.shape[1], Kc, Y_tr.shape[1], 1., links=True, gamma_param=True)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    losses = []
    tolerance = 1e-6
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tr)
        
        # Compute loss
        loss = 1-r2_torch(outputs, Y_tr)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())

        # Check convergence criterion
        if (len(losses) > 100) and (abs(1 - np.mean(losses[-101:-1])/(losses[-1]+1e-8)) < tolerance):
            break
    
    # Test prediction
    if -losses[-1] > loss_max:
        loss_max = -losses[-1]
        results['R2_tst'][i] = r2_torch(model(X_tst).detach().numpy(), Y_tst)
        results['componentes'][i] = Kc+D
        results['componentes_total'][i] = Kc+D
        results['componentes_lin'][i] = D
        results['componentes_rff'][i] = Kc

    return results, loss_max

def train_ELM_l(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):

    # Transforming to torch tensors
    X_tr = torch.from_numpy(X_tr).float()
    X_tst = torch.from_numpy(X_tst).float()
    Y_tr = torch.from_numpy(Y_tr).float()
    Y_tst = torch.from_numpy(Y_tst).float()

    D = X_tr.shape[1]

    # Initialize RVFL model
    model = RVFL(X_tr.shape[1], Kc, Y_tr.shape[1], 1./D, links=True, gamma_param=False)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    losses = []
    tolerance = 1e-6
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tr)
        
        # Compute loss
        loss = 1-r2_torch(outputs, Y_tr)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())

        # Check convergence criterion
        if (len(losses) > 100) and (abs(1 - np.mean(losses[-101:-1])/(losses[-1]+1e-8)) < tolerance):
            break
    
    # Test prediction
    if -losses[-1] > loss_max:
        loss_max = -losses[-1]
        results['R2_tst'][i] = r2_torch(model(X_tst).detach().numpy(), Y_tst)
        results['componentes'][i] = Kc+D
        results['componentes_total'][i] = Kc+D
        results['componentes_lin'][i] = D
        results['componentes_rff'][i] = Kc

    return results, loss_max

def train_ELM_g(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):

    # Transforming to torch tensors
    X_tr = torch.from_numpy(X_tr).float()
    X_tst = torch.from_numpy(X_tst).float()
    Y_tr = torch.from_numpy(Y_tr).float()
    Y_tst = torch.from_numpy(Y_tst).float()

    D = X_tr.shape[1]

    # Initialize RVFL model
    model = RVFL(X_tr.shape[1], Kc, Y_tr.shape[1], 1., gamma_param=True, links=False)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    losses = []
    tolerance = 1e-6
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tr)
        
        # Compute loss
        loss = 1-r2_torch(outputs, Y_tr)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())

        # Check convergence criterion
        if (len(losses) > 100) and (abs(1 - np.mean(losses[-101:-1])/(losses[-1]+1e-8)) < tolerance):
            break
    
    # Test prediction
    if -losses[-1] > loss_max:
        loss_max = -losses[-1]
        results['R2_tst'][i] = r2_torch(model(X_tst).detach().numpy(), Y_tst)
        results['componentes'][i] = Kc+D
        results['componentes_total'][i] = Kc+D
        results['componentes_lin'][i] = D
        results['componentes_rff'][i] = Kc

    return results, loss_max

def train_ELM(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):

    # Transforming to torch tensors
    X_tr = torch.from_numpy(X_tr).float()
    X_tst = torch.from_numpy(X_tst).float()
    Y_tr = torch.from_numpy(Y_tr).float()
    Y_tst = torch.from_numpy(Y_tst).float()

    D = X_tr.shape[1]

    # Initialize RVFL model
    model = RVFL(D, Kc, Y_tr.shape[1], 1./D, gamma_param=False, links=False)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    losses = []
    tolerance = 1e-6
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tr)
        
        # Compute loss
        loss = 1-r2_torch(outputs, Y_tr)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())

        # Check convergence criterion
        if (len(losses) > 100) and (abs(1 - np.mean(losses[-101:-1])/(losses[-1]+1e-8)) < tolerance):
            break
    
    # Test prediction
    if -losses[-1] > loss_max:
        loss_max = -losses[-1]
        results['R2_tst'][i] = r2_torch(model(X_tst).detach().numpy(), Y_tst)
        results['componentes'][i] = Kc+D
        results['componentes_total'][i] = Kc+D
        results['componentes_lin'][i] = D
        results['componentes_rff'][i] = Kc

    return results, loss_max

def train_ELM_bg(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):
    model = RFF_BLR()
    model.fit(X_tr, Y_tr, X_tst, Y_tst, 
              n_components = Kc, maxit = num_epochs, mult_gamma = False, links = False,
            gamma_param = True)

    if model.L[-1] > loss_max:
        results['R2_tst'][i] = model.R2_tst[-1]
        results['componentes'][i] = model.q_dist.K
        results['componentes_total'][i] = model.K
        results['componentes_lin'][i] = 0
        results['componentes_rff'][i] = model.q_dist.K
    return results, loss_max

def train_ELM_lbg(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):
    model = RFF_BLR()
    model.fit(X_tr, Y_tr, X_tst, Y_tst, 
              n_components = Kc, maxit = num_epochs, mult_gamma = False, links = True,
            gamma_param = True)
    if model.L[-1] > loss_max:
        results['R2_tst'][i] = model.R2_tst[-1]
        results['componentes'][i] = model.q_dist.K
        results['componentes_total'][i] = model.K
        results['componentes_lin'][i] = model.input_idx[Kc:].sum()
        results['componentes_rff'][i] = model.input_idx[:Kc].sum()
    return results, loss_max

def train_ELM_lb(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):
    D = X_tr.shape[1]
    model = RFF_BLR()
    model.fit(X_tr, Y_tr, X_tst, Y_tst, 
              n_components = Kc, maxit = num_epochs, mult_gamma = False, gamma = 1./D, links = True,
            gamma_param = False)
    if model.L[-1] > loss_max:
        results['R2_tst'][i] = model.R2_tst[-1]
        results['componentes'][i] = model.q_dist.K
        results['componentes_total'][i] = model.K
        results['componentes_lin'][i] = model.input_idx[Kc:].sum()
        results['componentes_rff'][i] = model.input_idx[:Kc].sum()
    return results, loss_max

def train_ELM_b(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i):    
    D = X_tr.shape[1]
    model = RFF_BLR()
    model.fit(X_tr, Y_tr, X_tst, Y_tst, 
              n_components = Kc, maxit = num_epochs, mult_gamma = False, gamma = 1./D, links = False,
            gamma_param = False)
    if model.L[-1] > loss_max:
        results['R2_tst'][i] = model.R2_tst[-1]
        results['componentes'][i] = model.q_dist.K
        results['componentes_total'][i] = model.K
        results['componentes_lin'][i] = model.input_idx[Kc:].sum()
        results['componentes_rff'][i] = model.input_idx[:Kc].sum()
    return results, loss_max

N, D, C = 500, 100, 16

from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state

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

X, Y = make_poly_regression(N, D, 
    n_targets=C,
    random_state=42,
    interaction_only=True,
    degree=4,
    include_bias=False,
    noise = 2.)

folds = 10
idx = np.random.randint(0,2,Y.shape[0]).astype(int)

from sklearn.model_selection import StratifiedKFold
skf_tst = StratifiedKFold(n_splits=folds, shuffle = True)
fold_tst = [f for  i, f in enumerate(skf_tst.split(X, idx))]
dict_fold_val = {}
for ii, f_tst in enumerate(fold_tst):
    pos_tr = f_tst[0]
    skf_val = StratifiedKFold(n_splits=folds, shuffle = True)
    fold_val = [f for  i, f in enumerate(skf_val.split(X[pos_tr], idx[pos_tr]))]
    dict_fold_val[ii] = fold_val


model_extensions = {
    'ELM': train_ELM,
    'ELM_l': train_ELM_l,
    'ELM_g': train_ELM_g,
    'ELM_b': train_ELM_b,
    'ELM_lg': train_ELM_lg,
    'ELM_lb': train_ELM_lb,
    'ELM_bg': train_ELM_bg,
    'ELM_lbg': train_ELM_lbg
}
# def ablation_study(name):
for name, function in model_extensions.items():
    from sklearn.preprocessing import StandardScaler

    filename = 'Results/Ablation/'+name+'_'+str(folds)+'folds.pkl'
    if os.path.exists(filename):
        print ("Loading existing model...")
        results = pickle.load(open(filename, "rb" ) )
    else:
        results = {'R2_tst': np.zeros((folds,)), 
                    'componentes': np.zeros((folds,)), 
                    'componentes_total': np.zeros((folds,)), 
                    'componentes_lin': np.zeros((folds,)), 
                    'componentes_rff': np.zeros((folds,))
                                }

    num_epochs = 5000
    print('Name training model '+name)
    for i in np.arange(len(fold_tst)):  
        print('---------> Fold '+str(i)+' <---------') 
        if results['R2_tst'][i] == 0:

            # Splitting the data into training and test sets.
            pos_tr = fold_tst[i][0]
            pos_tst =  fold_tst[i][1]
            
            Y_tr = Y[pos_tr]
            Y_tst = Y[pos_tst]
            X_tr = X[pos_tr,:]
            X_tst = X[pos_tst,:]
            
            # Normalizing the data
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_tst = scaler.transform(X_tst)

            Kc = int(X_tr.shape[0]*1.5)
            
            # if results['RVFL']['R2_tst'][i] == 0:
            import copy

            loss_max = -1e20
            for ini in np.arange(10):
                results, loss_max = function(X_tr, Y_tr, X_tst, Y_tst, Kc, num_epochs, results, loss_max, i)

            print('R2: %.2f, K: %d/%d' %(results['R2_tst'][i], results['componentes'][i], results['componentes_total'][i]))
            if os.path.exists(filename):
                results2 = pickle.load(open(filename, "rb" ) )
                results2['R2_tst'][i] = results['R2_tst'][i]
                results2['componentes'][i] = results['componentes'][i]    
                results2['componentes_total'][i] = results['componentes_total'][i]    
                results2['componentes_lin'][i] = results['componentes_lin'][i]    
                results2['componentes_rff'][i] = results['componentes_rff'][i]        
            else:
                results2 = copy.copy(results) 
            with open(filename, 'wb') as output:
                pickle.dump(results2, output, pickle.HIGHEST_PROTOCOL)            
    print('Final test R2  %.3f +/- %.3f, K %d +/- %d, K lin %d +/- %d, K rff %d +/- %d' %(
        np.mean(results['R2_tst']), np.std(results['R2_tst']),
        np.mean(results['componentes']), np.std(results['componentes']),
        np.mean(results['componentes_lin']), np.std(results['componentes_lin']),
        np.mean(results['componentes_rff']), np.std(results['componentes_rff'])))