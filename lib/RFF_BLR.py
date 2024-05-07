import numpy as np
from scipy import linalg
from sklearn.metrics import r2_score
import os
from RFF_optimizer import RFF_optimizer
import copy
from sklearn.utils import check_random_state

# Define probabilistic model in pytorch for gamma optimization
# So far we use an standar GP model, later we have to change the covariance (kernel) function

class RFF_BLR(object):
    def __init__(self):
        pass

    def fit(self, x, y, x_tst = None, y_tst = None, gamma = .5, n_components=1000, 
            random_state=1, mult_gamma = False, maxit = 200, prune = True, 
            perc = False, pruning_crit = 1e-1, verbose = False, tol = 1e-6, links = False,
            gamma_param = False):
        """Fit the model to the data.

        Parameters
        ----------
        x : numpy array
            Input data with shape NxD.
        y : numpy array
            Output data with shape NxD.
        x_tst : numpy array
            Input test data with shape N_tstxD.
        y_tst : numpy array
            Output test data with shape N_tstxD.
        gamma : float   
            Initial value of the gamma parameter.   
        n_components : int
            Number of random Fourier features.
        random_state : int
            Random seed for reproducibility.
        mult_gamma : bool   
            If True, the gamma parameter is a vector of length D.
        maxit : int
            Maximum number of iterations.
        prune : bool
            If True, prune the features.
        perc : bool
            If True, the pruning criteria is a percentage.
        pruning_crit : float
            Pruning criteria.
        verbose : bool
            If True, print the progress.
        tol : float
            Tolerance for convergence.
        links : bool
            If True, concatenate input with hidden layer output.
        gamma_param : bool
            If True, the gamma parameter is optimized.
        """

        self.x = x
        self.y = y.T  #(DxN)

        if y_tst is not None and x_tst is not None:
            self.tst = True
            self.x_tst = x_tst
            self.y_tst = y_tst.T  #(DxN_tst)
        else:
            self.tst = False
        self.links = links
        if self.links:
            self.K = n_components+x.shape[1] #num dimensiones input
        else:
            self.K = n_components
        self.D = y.shape[1] #num dimensiones output
        self.N = y.shape[0] # num datos 
     
        self.L = []
        self.mse = []
        self.mse_tst = []        
        self.rrmse_tst = []        
        self.R2 = []
        self.R2_tst = []
        self.K_vec = []
        self.input_idx = np.ones(self.K, bool)
        self.hyper = HyperParameters()
        # self.q_dist = Qdistribution(self.N, self.D, self.K, self.hyper)
        self.gamma = gamma
        self.gamma_param = gamma_param
        if self.gamma_param:
            print('\rInitialising...', end='\r', flush=True)
            LB = np.zeros(10)
            RFF_ini = []
            # Random weight initialization
            for i in range(len(LB)):
                self.q_dist = Qdistribution(self.N, self.D, self.K, self.hyper)
                RFF_ini.append(RFF_optimizer(x, self.y, gamma=self.gamma, n_components=n_components, random_state=random_state, mult_gamma=mult_gamma, links = self.links))

                self.z = RFF_ini[-1].get_RFF(x, self.input_idx).T #(KxN)
                if self.tst:
                    self.z_tst = RFF_ini[-1].get_RFF(x_tst, self.input_idx).T #(KxN_tst)
                
                self.ZZT = self.z @ self.z.T  #(KxK)
                self.YZT = y.T @ self.z.T  #(DxK) 
                
                self.update()
        
                RFF_ini[-1].sgd_step(self.q_dist.W['mean'], self.q_dist.W['prodT'],
                                        self.q_dist.tau_mean(), self.q_dist.b['mean'], 
                                        1, self.input_idx)
                
                LB[i] = -RFF_ini[-1].L.item()
            self.q_dist = Qdistribution(self.N, self.D, self.K, self.hyper, y = self.y, z = self.z)
            # We start computing RFF 
            self.RFF_model = RFF_ini[np.argmax(LB)]
            self.z = self.RFF_model.get_RFF(x, self.input_idx).T #(KxN)
            if self.tst:
                self.z_tst = self.RFF_model.get_RFF(x_tst, self.input_idx).T #(KxN_tst)
        else:
            random_state = check_random_state(random_state)
            self.random_weights_unscaled_ = random_state.normal(size=(x.shape[1], n_components))
            self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=n_components)
            self.z = self.computeRFF(x, self.input_idx).T #(KxN)
            if self.tst:
                self.z_tst = self.computeRFF(x_tst, self.input_idx).T #(KxN_tst)
            self.q_dist = Qdistribution(self.N, self.D, self.K, self.hyper, 
                                        y = self.y, z = self.z)
        
        # Some precomputed matrices
        self.ZZT = self.z @ self.z.T  #(KxK)
        self.YZT = y.T @ self.z.T  #(DxK) 
        self.fit_vb(maxit, pruning_crit, prune, tol, perc, verbose)
            
    def pruning(self, pruning_crit, perc = False):       
        q = self.q_dist
        if perc:
            fact_sel = np.argsort(np.mean(abs(q.W['mean']),axis=0))[:int(q.W['mean'].shape[1]*pruning_crit)]
        else:
            fact_sel = np.array([])
            for K in np.arange(q.K):
                if any(abs(q.W['mean'][:,K]) > pruning_crit):
                    fact_sel = np.append(fact_sel,K)
            fact_sel = np.unique(fact_sel).astype(int)
        
        aux = self.input_idx[self.input_idx]
        aux[fact_sel] = False
        self.input_idx[self.input_idx] = ~aux
        
        # Pruning W and alpha
        q.W['mean'] = q.W['mean'][:,fact_sel]
        q.W['cov'] = q.W['cov'][fact_sel,:][:,fact_sel]
        q.W['prodT'] = q.W['prodT'][fact_sel,:][:,fact_sel]
        self.z = self.z[fact_sel,:]
        if self.tst:
          self.z_tst = self.z_tst[fact_sel,:]
        self.ZZT = self.ZZT[fact_sel,:][:,fact_sel]
        self.YZT = self.YZT[:,fact_sel]
        q.alpha['b'] = q.alpha['b'][fact_sel]
        q.K = len(fact_sel)

    def computeRFF(self, X, input_idx):        
        n_components = len(self.random_offset_)
        self.random_weights_ = np.sqrt(2 * self.gamma) * self.random_weights_unscaled_[:,input_idx[:n_components]]
        X_rff = np.cos(X @ self.random_weights_ + self.random_offset_[input_idx[:n_components]]) * np.sqrt(2./n_components)
        return np.hstack((X_rff, X[:,input_idx[n_components:]]))
    
    def compute_rrmse(self, z, y):
        if z is None:
            z = self.z
        if y is None:
            y = self.y
        nume = np.sum((self.predict(z) - y)**2.0)
        deno = np.sum((np.outer(np.mean(self.y,1), np.ones(y.shape[1])) - y)**2.0)
        return np.mean(np.sqrt(nume/deno))

    def compute_mse(self, z = None, y = None):
        q = self.q_dist
        if z is None:
            z = self.z
        if y is None:
            y = self.y
        diff = (y - q.W['mean'] @ z).ravel()
        return  diff@diff/z.shape[1]
    
    def compute_R2(self, z = None, y = None):
        q = self.q_dist
        if z is None:
            z = self.z
        if y is None:
            y = self.y
        return  r2_score(y.ravel(), (self.predict(z)).ravel())
        
    def predict(self, z):
        q = self.q_dist
        return q.W['mean'] @ z + q.b['mean']
        
    def fit_vb(self, maxit=200, pruning_crit = 1e-1, prune = True, tol = 1e-6, perc = False, verbose = False):
        """
        Fit the model using Variational Bayes.
        
        Parameters
        ----------
        maxit : int
            Maximum number of iterations.
        pruning_crit : float
            Pruning criteria.
        prune : bool
            If True, prune the features.
        tol : float
            Tolerance for convergence.
        perc : bool
            If True, the pruning criteria is a percentage.
        verbose : bool
            If True, print the progress.
        """
        q = self.q_dist
        for i in range(maxit):
            self.update()
            self.mse.append(self.compute_mse())
            self.R2.append(self.compute_R2())
            if self.tst:
              self.mse_tst.append(self.compute_mse(self.z_tst, self.y_tst))
              self.rrmse_tst.append(self.compute_rrmse(self.z_tst, self.y_tst))
              self.R2_tst.append(self.compute_R2(self.z_tst, self.y_tst))
            self.K_vec.append(q.K)
            # self.depruning(1e-15)
            self.L.append(self.update_bound())

            if prune:
                self.pruning(pruning_crit, perc)
            if q.K == 0:
                print('\nThere are no representative features, no structure found in the data. Try changing the pruning criteria')
                return
            if verbose:
                print('\rIteration %d K %4d' %(i+1, q.K), end='\r', flush=True)
            if (len(self.L) > 100) and (abs(1 - np.mean(self.L[-101:-1])/self.L[-1]) < tol):
                # print('\nModel correctly trained. Convergence achieved')
                return
        if verbose:
            print('')

    def update(self):
        """
        Update the model.
        """
        self.update_b()
        self.update_w()
        self.update_alpha()
        self.update_tau()
        if self.gamma_param and len(self.L) % 10 == 0 and len(self.L) > 0:
            self.RFF_model.sgd_step(self.q_dist.W['mean'], self.q_dist.W['prodT'], 
                                    self.q_dist.tau_mean(), self.q_dist.b['mean'], 
                                    50, self.input_idx)
            self.update_kernel()
    
    def myInverse(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        try:
            L = linalg.pinv(np.linalg.cholesky(X), rcond=1e-10) #np.linalg.cholesky(A)
            return np.dot(L.T,L) #linalg.pinv(L)*linalg.pinv(L.T)
        except:
            return np.nan
        
    def update_w(self):
        """Updates the variable W.

        This function uses the variables of the learnt model to update W.

        """
        q = self.q_dist
        # cov
        w_cov = self.myInverse(np.diag(q.alpha_mean()) + q.tau_mean() * self.ZZT)
        
        if not np.any(np.isnan(w_cov)):
            q.W['cov'] = w_cov
            # mean
            q.W['mean'] = q.tau_mean() * (np.subtract(self.y, q.b['mean'])) @ self.z.T @ q.W['cov']
            #E[W*W^T]
            q.W['prodT'] = q.W['mean'].T @ q.W['mean'] + self.D*q.W['cov']
            
        # else:
        #     print ('Cov W is not invertible, not updated')

    def update_b(self):
        """Updates the variable b.
        
        This function uses the variables of the learnt model to update b.
        """
        q = self.q_dist
        q.b['cov'] = (1 + self.N * q.tau_mean())**(-1) * np.eye(self.D)
        q.b['mean'] = q.tau_mean() * np.dot(q.b['cov'],
            np.sum(np.subtract(self.y, np.dot(q.W['mean'], self.z)), axis=1)[:,np.newaxis])
        q.b['prodT'] = np.sum(q.b['mean']**2) + self.D*q.b['cov'][0,0]    #mean of a noncentral chi-squared distribution

    def update_alpha(self):
        """Updates the variable alpha.
        
        This function uses the variables of the learnt model to update alpha.
        """
        q = self.q_dist
        q.alpha['a'] = self.hyper.alpha_a/(self.D) + 0.5
        q.alpha['b'] = (self.hyper.alpha_b + 0.5 * np.diag(q.W['prodT']))/(self.D)
        
    def update_tau(self):
        """Updates the variable tau.

        This function uses the variables of the learnt model to update tau.
        """
        q = self.q_dist
        q.tau['a'] = self.hyper.tau_a/(self.D*self.N) + 0.5
        q.tau['b'] = (self.hyper.tau_b + 0.5 *(np.sum(self.y.ravel()**2)
                                               + np.einsum('ij,ji->', q.W['prodT'], self.ZZT) 
                                               + self.N * q.b['prodT'] 
                                               - 2 * np.einsum('ij,ji->', q.W['mean'], self.YZT.T)
                                               + 2 * np.sum(q.b['mean'].T @ q.W['mean'] @ self.z) 
                                               - 2 * np.sum(q.b['mean'].T @ self.y)
                                               ))/(self.D*self.N)

    def update_kernel(self):
        """Updates the kernel.

        This function uses the variables of the learnt model to update the kernel.
        """
        self.z = self.RFF_model.get_RFF(self.x, self.input_idx).T #(KxN)
        if self.tst:
          self.z_tst = self.RFF_model.get_RFF(self.x_tst, self.input_idx).T #(KxN_tst)
        # Some precomputed matrices
        self.ZZT = self.z @ self.z.T  #(KxK) es enorme, habría que ver si se puede evitar este calculo
        self.YZT = self.y @ self.z.T  #(DxK)

    def HGamma(self, a, b):
        """Compute the entropy of a Gamma distribution.

        Parameters
        ----------
        __a: float. 
            The parameter a of a Gamma distribution.
        __b: float. 
            The parameter b of a Gamma distribution.

        """
        
        return -np.log(b+1e-8)
    
    def HGauss(self, mn, cov, entr):
        """Compute the entropy of a Gamma distribution.
        
        Uses slogdet function to avoid numeric problems. If there is any 
        infinity, doesn't update the entropy.
        
        Parameters
        ----------
        __mean: float. 
            The parameter mean of a Gamma distribution.
        __covariance: float. 
            The parameter covariance of a Gamma distribution.
        __entropy: float.
            The entropy of the previous update. 

        """
        
        H = 0.5*mn.shape[0]*np.linalg.slogdet(cov)[1]
        return self.checkInfinity(H, entr)
        
    def checkInfinity(self, H, entr):
        """Checks if three is any infinity in th entropy.
        
        Goes through the input matrix H and checks if there is any infinity.
        If there is it is not updated, if there isn't it is.
        
        Parameters
        ----------
        __entropy: float.
            The entropy of the previous update. 

        """
        
        if abs(H) == np.inf:
            return entr
        else:
            return H
        
    def update_bound(self):
        """Update the Lower Bound.
        
        Uses the learnt variables of the model to update the lower bound.
        """
        
        q = self.q_dist
        
        q.W['LH'] = self.HGauss(q.W['mean'], q.W['cov'], q.W['LH'])
        q.b['LH'] = self.HGauss(q.b['mean'], q.b['cov'], q.b['LH'])
        # Entropy of alpha and tau
        q.alpha['LH'] = np.sum(self.HGamma(q.alpha['a'], q.alpha['b']))
        q.tau['LH'] = np.sum(self.HGamma(q.tau['a'], q.tau['b']))
            
        # Total entropy
        EntropyQ = q.W['LH'] + q.b['LH'] + q.alpha['LH']  + q.tau['LH']
        
        # Calculation of the E[log(p(Theta))]
        q.tau['ElogpXtau'] = -(0.5 *  self.D + self.hyper.tau_a - 1)* np.log(q.tau['b']+1e-8)
        q.alpha['ElogpWalp'] = -(0.5 * self.D + self.hyper.alpha_a - 1)* np.sum(np.log(q.alpha['b']+1e-8))
        q.b['Elogp'] = -0.5*q.b['prodT']
        
        # Total E[log(p(Theta))]
        ElogP = q.tau['ElogpXtau'] + q.alpha['ElogpWalp'] +q.b['Elogp']
        return ElogP - EntropyQ

class HyperParameters(object):

    def __init__(self):
        self.alpha_a = 1e-13
        self.alpha_b = 1e-14
        self.tau_a = 1e-14
        self.tau_b = 1e-14
            
class Qdistribution(object):
    def __init__(self, N, D, K, hyper, y = [None], z = [None]):
        self.N = N
        self.D = D
        self.K = K
        
        # Initialize gamma disributions
        alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.K)
        self.alpha = alpha 
        tau = self.qGamma(hyper.tau_a,hyper.tau_b,1)
        self.tau = tau 

        # The remaning parameters at random
        self.init_rnd(y, z)

    def init_rnd(self, y, z):
        self.W = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        self.b = copy.deepcopy(self.W)
        
        self.W["mean"] = np.random.normal(0.0, 1.0, self.D * self.K).reshape(self.D, self.K)
        self.W["cov"] = np.eye(self.K)
        self.W["prodT"] = np.dot(self.W["mean"].T, self.W["mean"])+self.K*self.W["cov"]
        
        if None in y:
            self.b["mean"] = np.random.normal(0.0, 1.0, self.D).reshape(self.D, 1)
            self.b["cov"] = np.eye(self.D)
        else:
            self.b['cov'] = (1 + self.N * self.tau_mean())**(-1) * np.eye(self.D)
            self.b['mean'] = self.tau_mean() * np.dot(self.b['cov'],
                np.sum(np.subtract(y, np.dot(self.W['mean'], z)), axis=1)[:,np.newaxis])
        self.b['prodT'] = np.sum(self.b['mean']**2) + self.D*self.b['cov'][0,0]    #mean of a noncentral chi-squared distribution

    def qGamma(self,a,b,K):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [1, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [K, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __K: array (shape = [K, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = {                
                "a":         a,
                "b":         (b*np.ones((K,1))).flatten(),
                "LH":         None,
                "ElogpWalp":  None,
            }
        return param
        
    def alpha_mean(self):
        return self.alpha['a'] / self.alpha['b']
    
    def tau_mean(self):
        return self.tau['a'] / self.tau['b']
    
