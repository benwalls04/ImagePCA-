import numpy as np
import matplotlib.pyplot as plt
import math
import csv 


def PCA(X, out_dim):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        out_dim: the desired output dimension
    Output:
        mu: the mean vector of X. Please represent it as a D-by-1 matrix (numpy array).
        W: the projection matrix of PCA. Please represent it as a D-by-out_dim matrix (numpy array).
            The m-th column should correspond to the m-th largest eigenvalue of the covariance matrix.
            Each column of W must have a unit L2 norm.
    Todo:
        1. build mu
        2. build the covariance matrix Sigma: a D-by-D matrix (numpy array).
        3. We have provided code of how to compute W from Sigma
    Useful tool:
        1. np.mean: find the mean vector
        2. np.matmul: for matrix-matrix multiplication
        3. the builtin "reshape" and "transpose()" function of a numpy array
    """

    X = np.copy(X) # copy of the input data (avoid aliasing)
    D = X.shape[0] # feature dimension
    N = X.shape[1] # number of data instances

    ### Your job  starts here ###
    """
        use the following:
        np.linalg.eigh (or np.linalg.eig) for eigendecomposition. it returns
        V: eigenvalues, W: eigenvectors
        This function has already L2 normalized each eigenvector.
        NOYE: the output may be complex value: do .real to keep the real part of V and W
        sort the eigenvectors by sorting corresponding eigenvalues
        return mu and W
    
    """
    mu = np.mean(X, axis=1, keepdims=True)

    # compute the covariance matrix 
    X_centered = X - mu
    Cov = X_centered @ X_centered.T / N
    
    # compute the eigendecomposition 
    evals, evecs = np.linalg.eig(Cov)
    evals = evals.real
    evecs = evecs.real

    # sort the evecs by corresponsding evals. Select the top M evecs as components 
    sort_evals = np.argsort(evals)[::-1] 
    W = evecs[:, sort_evals][:, :out_dim]
    W = W[:, :out_dim]

    return mu, W

    ### Your job  ends here ###


### Your job  starts here ###   

def reconstruct(z, W, X_mean):
    return  W @ z + X_mean

def mse_error(X, X_reconstructed):
    # Compute the squared error matrix 
    squared_errors = (X - X_reconstructed) ** 2
    
    # average over resulting matrix 
    mse = np.mean(squared_errors)
    
    return mse

def step_PCA(X, test_set, start_dim, end_dim, step_size):
    errors = []
    
    # Compute covariance matrix of test_set instead of X
    mu_test = np.mean(test_set, axis=1, keepdims=True)
    test_centered = test_set - mu_test

    for dim in range(start_dim, end_dim + 1, step_size):
        # Compute PCA projection using full dataset
        mu, W = PCA(X, dim)  

        # compute PCA of test set using components from the full dataset 
        z = W.T @ test_centered  
        test_reconstructed = reconstruct(z, W, mu_test)  
        error = mse_error(test_set, test_reconstructed)

        # print the error to the console every 100 dimensions (for progress tracking)
        if dim % 100 == 0: 
            print(f"Dimension: {dim}, Error: {error}")

        errors.append(error)
    
    return errors


"""
    load MNIST
    compute PCA
    produce figures and plots
"""

# load dataset and seperate labels from image vectors 
data = np.loadtxt("mnist_test.csv", delimiter=",").T
labels = data[0, :]
entries = data[1:, :]

# select all the vectors where corresponding label is 3
threes = entries[:, labels == 3]
X = threes[:, 0].reshape(-1, 1) 


plt.figure(figsize=(10, 6))

# plot the original image 
plt.subplot(1, 6, 1)
plt.imshow(X.reshape(28, 28))
plt.title("Original")
plt.xticks([])  
plt.yticks([])  

dims = [2, 8, 64, 128, 784]
for i, dim in enumerate(dims):
    # compute PCA using each dimension 
    mu, W = PCA(threes, dim)
    X_centered = X - mu
    z = W.T @ X_centered  
    X_reconstructed = reconstruct(z, W, mu)  

    # plot the reconstructed image 
    plt.subplot(1, 6, i + 2)
    plt.imshow(X_reconstructed.reshape(28, 28))
    plt.title(f"({dim} PCs)")
    plt.xticks([])  
    plt.yticks([])  

# show the 5 images in one plot 
plt.tight_layout()
plt.show()

# split the threes into test and train sets
test_set = threes[:, :100]
threes_train = threes[:, 100:]

# make other needed training sets
train_set2 = np.concatenate((threes_train, entries[:, (labels == 8)]), axis=1)
train_set3 = np.concatenate((train_set2, entries[:, (labels == 9)]), axis=1)

# define the start dimension, end dimension, and step size (hyperparameters)
SD, ED, SS = 10, 784, 10
all_errors = []

# compute the errors for each dataset 
all_errors.append(step_PCA(threes_train, test_set, SD, ED, SS))
all_errors.append(step_PCA(train_set2, test_set, SD, ED, SS))
all_errors.append(step_PCA(train_set3, test_set, SD, ED, SS))

# get the dimensions used for the x axis  
dimensions = list(range(SD, ED + 1, SS))

# plot the errors for each dataset on the same splot 
plt.figure(figsize=(10, 6))
plt.plot(dimensions, all_errors[0], label='Threes')
plt.plot(dimensions, all_errors[1], label='Threes and Eights')
plt.plot(dimensions, all_errors[2], label='Threes, Eights, and Nines')

# show the plot 
plt.xlabel('Dimension')
plt.ylabel('Error')
plt.title('PCA Reconstruction Error vs. Dimension')
plt.legend()
plt.show()








    



    






