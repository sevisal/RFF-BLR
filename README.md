# RFF-BLR: Random Fourier Features Bayesian Linear Regression

Welcome to the RFF-BLR repository! This repository houses the implementation of the Random Fourier Features Bayesian Linear Regression (RFF-BLR) presented in the paper [Bayesian learning of feature spaces for multitask problems](https://arxiv.org/abs/2209.03028). 

## Overview

In this repository, we delve into the world of advanced regression techniques with a focus on Bayesian inference and the Random Fourier Features. We provide an in-depth exploration through an ablation study, comparing various model extensions such as Bayesian formulation, Links, and Gamma optimization. Additionally, we conduct a rigorous comparison against different baselines using synthetic datasets, shedding light on the profound impact of RFF and its interaction with Bayesian Sparse formulations. Lastly, we offer a computational cost analysis, illuminating the efficiency of the RFF-BLR model in diverse scenarios using synthetic datasets.

## Installation

To get started, follow these simple installation steps:

```bash
# Clone the repository
git clone https://github.com/sevisal/RFF-BLR.git

# Navigate into the cloned directory
cd RFF-BLR

# Install the required Python packages
pip install -r requirements.txt
```
## Usage

This repository offers various scripts designed for different aspects of the study:

1. **Baseline Comparison**: Run the `BaselineComparison.py` script to initiate the baseline comparison:

    ```bash
    python BaselineComparison.py
    ```

    Witness the comparative performance of the RFF-BLR model against other baseline models using synthetic datasets.

2. **Ablation Study**: Execute the `AblationStudy.py` script to run the ablation study:

    ```bash
    python AblationStudy.py
    ```

    Dive into the realm of model extensions with functions meticulously crafted for each combination, exploring Bayesian formulations with gamma optimization or  with RVFL-like links.

3. **Computational Cost Analysis**: To run the computational cost analysis, execute the `CostAnalysis.py` script:

    ```bash
    python CostAnalysis.py
    ```

    Experience a detailed analysis of computational costs, providing invaluable insights into the performance of the RFF-BLR model using synthetic datasets.