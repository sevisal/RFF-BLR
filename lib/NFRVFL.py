import numpy as np
from scipy.special import expit as sigmoid
from sklearn.cluster import KMeans
import time

class NFRVFL:
    """
    Neuro-Fuzzy Regularized Random Vector Functional Link (Neuro-Fuzzy RVFL) Classifier

    Parameters:
    C: Regularization Parameter
    NumFuzzyRule: Number of Fuzzy Rules
    NumHiddenNodes: Number of Hidden Nodes
    Alpha: Input to Hidden Layer Weights
    WeightHidden: Hidden to Output Layer Weights
    activation: Activation Function (1: Sigmoid, 2: Sin, 3: ReLU)
    clus: Clustering Method (1: KMeans, 2: Random)
    """
    def __init__(self, C, NumFuzzyRule, NumHiddenNodes, Alpha = None, WeightHidden = None, activation = 1, clus = 1):
        self.Alpha = Alpha
        self.WeightHidden = WeightHidden
        self.C = C
        self.NumFuzzyRule = NumFuzzyRule
        self.NumHiddenNodes = NumHiddenNodes
        self.activation = activation
        self.clus = clus
        self.std = 1
        self.beta = None
        self.center = None

    def forward(self, X):
        Omega = np.zeros((X.shape[0], self.NumFuzzyRule))
        F = np.zeros((X.shape[0], self.NumFuzzyRule))  # Fuzzy Layer

        # Calculating fuzzy membership value
        for j in range(X.shape[0]):
            MF = np.exp(-(np.tile(X[j, :], (self.NumFuzzyRule, 1)) - self.center)**2 / self.std)
            MF = np.prod(MF, axis=1)
            MF = MF / np.sum(MF + 1e-10)
            F[j, :] = MF * (X[j, :] @ self.Alpha)
            Omega[j, :] = MF

        F1 = np.hstack([F, 0.1 * np.ones((F.shape[0], 1))])
        H = F1 @ self.WeightHidden

        if self.activation == 1:
            H = sigmoid(H)
        elif self.activation == 2:
            H = np.sin(H)
        elif self.activation == 3:
            H = np.maximum(0, H)  # ReLU activation
        else:
            print('Activation Function Not Found. Please Choose from 1, 2, or 3. Defaulting to Sigmoid.')
            H = sigmoid(H)
        
        H = np.hstack([H, X])  # Direct Link
        M = np.hstack([Omega * (X @ self.Alpha), H])

        return M

    def fit(self, X, Y):
        """
        Training the Neuro-Fuzzy RVFL
        
        Parameters:
        X: Input Data
        Y: Output Data
        """
        
        # Randomly initializing parameters for the fuzzy layer and hidden layer
        self.Alpha = np.random.rand(X.shape[1], self.NumFuzzyRule) if self.Alpha is None else self.Alpha  
        self.WeightHidden = np.random.rand(self.NumFuzzyRule + 1, self.NumHiddenNodes) if self.WeightHidden is None else self.WeightHidden  

        # Clustering Methods
        if self.clus == 1:
            kmeans = KMeans(n_clusters=self.NumFuzzyRule, random_state=0, n_init = 'auto').fit(X)
            self.center = kmeans.cluster_centers_
        else:
            Temptrain_x = np.random.permutation(len(X))
            indices = Temptrain_x[:self.NumFuzzyRule]
            self.center = X[indices, :]

        M = self.forward(X)
        Nsample, _ = X.shape

        # Finding Output Layer Parameter (Here, beta)
        if M.shape[1] < Nsample:
            self.beta = np.linalg.solve(M.T @ M + np.eye(M.shape[1]) * (1 / self.C), M.T @ Y)
        else:
            self.beta = M.T @ np.linalg.solve(np.eye(M.shape[0]) * (1 / self.C) + M @ M.T, Y)

    def predict(self, X):
        """
        Predicting the Output
        
        Parameters:
        X: Input Data
        """
        # Testing starts
        M1 = self.forward(X)
        PredictedTestLabel = M1 @ self.beta  # Test Prediction

        return PredictedTestLabel