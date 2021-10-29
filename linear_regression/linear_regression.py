import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Dict, Tuple

# prep phase - load and prepare boston data from sklearn
def prep_phase_boston() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
  
  #1: load data
  from sklearn.datasets import load_boston
  boston = load_boston()
  data = boston.data
  target = boston.target
  features = boston.feature_names

  #2: scale data
  from sklearn.preprocessing import StandardScaler
  s = StandardScaler()
  data = s.fit_transform(data)

  #3: create train/test split
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state = 80718)
  y_train = y_train.reshape(-1, 1)
  y_test = y_test.reshape(-1, 1)

  return X_train, X_test, y_train, y_test


def forward_linear_regression(X_batch: ndarray,
                              y_batch: ndarray,
                              weights: Dict[str, ndarray]
                              )-> Tuple[float, Dict[str, ndarray]]:
    '''
    Forward pass for the step-by-step linear regression.
    '''
    # assert batch sizes of X and y are equal
    assert X_batch.shape[0] == y_batch.shape[0]

    # assert that matrix multiplication can work
    assert X_batch.shape[1] == weights['W'].shape[0]

    # assert that B is simply a 1x1 ndarray
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    # compute the operations on the forward pass
    N = np.dot(X_batch, weights['W'])

    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    # save the information computed on the forward pass
    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info

def loss_gradients(forward_info: Dict[str, ndarray], weights: Dict[str, ndarray]) -> Dict[str, ndarray]:

  y = forward_info['y']
  P = forward_info['P']
  N = forward_info['N']
  X = forward_info['X']
  B = weights['B']

  # derivative of (y - P)² = -1 * 2(y - P) = -2 * (y - P)
  dLdP = -2 * (y - P)

  dPdN = np.ones_like(N)

  dPdB = np.ones_like(B)

  dLdN = dLdP * dPdN
  
  dNdW = np.transpose(X, (1, 0))

  # Derivate from loss to weights
  dLdW = np.dot(dNdW, dLdN)

  # Derivate from Loss to bias
  dLdB = (dLdP * dPdB).sum(axis=0)
  
  loss_gradients_result: Dict[str, ndarray] = {}
  loss_gradients_result['W'] = dLdW
  loss_gradients_result['B'] = dLdB
  
  return loss_gradients_result

# helper functions and types
Batch = Tuple[ndarray, ndarray]

def generate_batch(X: ndarray, 
                   y: ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:
    '''
    Generate batch from X and y, given a start position
    '''
    assert X.ndim == y.ndim == 2, \
    "X and Y must be 2 dimensional"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start
    
    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
    
    return X_batch, y_batch

def init_weights(n_in: int) -> Dict[str, ndarray]:
    '''
    Initialize weights on first forward pass of model.
    '''
    
    weights: Dict[str, ndarray] = {}
    W = np.random.randn(n_in, 1)
    B = np.random.randn(1, 1)
    
    weights['W'] = W
    weights['B'] = B

    return weights

def permute_data(X: ndarray, y: ndarray):
    '''
    Permute (Shuffle) X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


# training functions
def train_one_step(X_batch: ndarray, y_batch: ndarray, weights: Dict[str, ndarray], learning_rate: float = 0.01) -> Dict[str, ndarray]:
  loss, forward_info = forward_linear_regression(X_batch, y_batch, weights)
  loss_grads = loss_gradients(forward_info, weights)

  for key in loss_grads.keys():
    weights[key] -= learning_rate * loss_grads[key]

  return loss, weights

def train(X: ndarray, 
          y: ndarray, 
          epochs: int = 1000, 
          learning_rate: float = 0.01,
          batch_size: int = 100, 
          seed: int = 1) -> Dict[str, ndarray]:

  assert epochs > 0

  if seed:
    np.random.seed(seed)
  start = 0

  weights = init_weights(X.shape[1])

  X, y = permute_data(X, y)

  losses = []

  for i in range(epochs):

    if start >= X.shape[0]:
      X, y = permute_data(X, y)
      start = 0

    X_batch, y_batch = generate_batch(X, y, start, batch_size)
    start += batch_size

    loss, weights = train_one_step(X_batch, 
                                   y_batch, 
                                   weights,
                                   learning_rate)
    losses.append(loss)
    print("Successfully trained step: " + str(i) + "\r", end="")
  
  return losses, weights

# prediction
def predict(X: ndarray, weights: Dict[str, ndarray]) -> ndarray:
  N = np.dot(X, weights['W'])

  return N + weights['B']

def main():
  X_train, X_test, y_train, y_test = prep_phase_boston()

  epochs = 1000

  losses, weights = train(X=X_train, 
                          y=y_train, 
                          batch_size=23, 
                          epochs=epochs,
                          learning_rate=0.001,
                          seed=180708)
  # plt.plot(list(range(epochs)), losses)
  # plt.show()

  preds = predict(X_test, weights)
  mae = np.mean(np.abs(y_test - preds))
  rmse = np.sqrt(np.mean(np.power(y_test - preds, 2)))


  maximum = max(np.max(preds), np.max(y_test))

  # plt.xlabel("Predicted value")
  # plt.ylabel("Actual value")
  # plt.title("Predicted vs. Actual values for\ncustom linear regression model");
  # plt.xlim([0, np.max(preds) + 1])
  # plt.ylim([0, np.max(y_test) + 1])
  # plt.scatter(preds, y_test)
  # plt.plot([0, maximum], [0, maximum]);
  # plt.show()
  print("Mean absolute error: " + str(mae))
  print("Root mean squared error: " + str(rmse))

main()