import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Dict, Tuple
from sklearn.metrics import r2_score

Batch = Tuple[ndarray, ndarray]

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


# helper functions
def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-1.0 * x))

def init_weights(input_size: int, 
                 hidden_size: int) -> Dict[str, ndarray]:
    '''
    Initialize weights during the forward pass for step-by-step neural network model.
    '''
    weights: Dict[str, ndarray] = {}
    weights['W1'] = np.random.randn(input_size, hidden_size)
    weights['B1'] = np.random.randn(1, hidden_size)
    weights['W2'] = np.random.randn(hidden_size, 1)
    weights['B2'] = np.random.randn(1, 1)
    return weights

def permute_data(X: ndarray, y: ndarray):
    '''
    Permute (Shuffle) X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

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


# the network
def forward_loss(X: ndarray,
                 y: ndarray,
                 weights: Dict[str, ndarray]
                 ) -> Tuple[Dict[str, ndarray], float]:
    '''
    Compute the forward pass and the loss for the step-by-step 
    neural network model.     
    '''
    M1 = np.dot(X, weights['W1'])

    N1 = M1 + weights['B1']

    O1 = sigmoid(N1)
    
    M2 = np.dot(O1, weights['W2'])

    P = M2 + weights['B2']    

    loss = np.mean(np.power(y - P, 2))

    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P'] = P
    forward_info['y'] = y

    return forward_info, loss

def loss_gradients(forward_info: Dict[str, ndarray], 
                   weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    '''
    Compute the partial derivatives of the loss with respect to each of the parameters in the neural network.
    '''    
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    
    dPdM2 = np.ones_like(forward_info['M2'])

    dLdM2 = dLdP * dPdM2
  
    dPdB2 = np.ones_like(weights['B2'])

    dLdB2 = (dLdP * dPdB2).sum(axis=0)
    
    dM2dW2 = np.transpose(forward_info['O1'], (1, 0))
    
    dLdW2 = np.dot(dM2dW2, dLdP)

    dM2dO1 = np.transpose(weights['W2'], (1, 0)) 

    dLdO1 = np.dot(dLdM2, dM2dO1)
    
    dO1dN1 = sigmoid(forward_info['N1']) * (1- sigmoid(forward_info['N1']))
    
    dLdN1 = dLdO1 * dO1dN1
    
    dN1dB1 = np.ones_like(weights['B1'])
    
    dN1dM1 = np.ones_like(forward_info['M1'])
    
    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)
    
    dLdM1 = dLdN1 * dN1dM1
    
    dM1dW1 = np.transpose(forward_info['X'], (1, 0)) 

    dLdW1 = np.dot(dM1dW1, dLdM1)

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W2'] = dLdW2
    loss_gradients['B2'] = dLdB2.sum(axis=0)
    loss_gradients['W1'] = dLdW1
    loss_gradients['B1'] = dLdB1.sum(axis=0)
    
    return loss_gradients

# training functions
def train(X_train: ndarray, y_train: ndarray,
          X_test: ndarray, y_test: ndarray,
          n_iter: int = 1000,
          test_every: int = 1000,
          learning_rate: float = 0.01,
          hidden_size= 13,
          batch_size: int = 100,
          return_losses: bool = True, 
          return_weights: bool = True, 
          return_scores: bool = True,
          seed: int = 1) -> None:

  if seed:
      np.random.seed(seed)

  start = 0

  # Initialize weights
  weights = init_weights(X_train.shape[1], 
                          hidden_size=hidden_size)

  # Permute data
  X_train, y_train = permute_data(X_train, y_train)
  

  losses = []
      
  val_scores = []

  for i in range(n_iter):

    # Generate batch
    if start >= X_train.shape[0]:
        X_train, y_train = permute_data(X_train, y_train)
        start = 0
    
    X_batch, y_batch = generate_batch(X_train, y_train, start, batch_size)
    start += batch_size

    # Train net using generated batch
    forward_info, loss = forward_loss(X_batch, y_batch, weights)

    if return_losses:
        losses.append(loss)

    loss_grads = loss_gradients(forward_info, weights)
    for key in weights.keys():
        weights[key] -= learning_rate * loss_grads[key]
    
    if return_scores:
        if i % test_every == 0 and i != 0:
            preds = predict(X_test, weights)
            val_scores.append(r2_score(preds, y_test))

  if return_weights:
      return losses, weights, val_scores
  
  return None

#prediction
def predict(X: ndarray, weights: Dict[str, ndarray]) -> float:
  '''
  Compute the forward pass and the loss for the step-by-step 
  neural network model.     
  '''
  M1 = np.dot(X, weights['W1'])
  N1 = M1 + weights['B1']
  O1 = sigmoid(N1)
  M2 = np.dot(O1, weights['W2'])
  P = M2 + weights['B2']    

  return P

#run the stuff
def main():
  X_train, X_test, y_train, y_test = prep_phase_boston()

  epochs = 10000
  test_every = 1000
  train_info = train(X_train, y_train, X_test, y_test,
                      n_iter=epochs,
                      test_every = test_every,
                      learning_rate = 0.001,
                      batch_size=23, 
                      return_losses=True, 
                      return_weights=True, 
                      return_scores=True,
                      seed=80718)
  losses = train_info[0]
  weights = train_info[1]
  val_scores = train_info[2]

  plt.ylim([-1,1])
  plt.plot(list(range(int(epochs / test_every - 1))), val_scores); 
  plt.xlabel("Batches (000s)")
  plt.title("Validation Scores")
  plt.show()

  plt.plot(list(range(10000)), losses);
  plt.show()

  preds = predict(X_test, weights)
  mae = np.mean(np.abs(y_test - preds))
  rmse = np.sqrt(np.mean(np.power(y_test - preds, 2)))
  print("Mean absolute error: " + str(mae))
  print("Root mean squared error: " + str(rmse))

  maximum = max(np.max(preds), np.max(y_test))

  plt.xlabel("Predicted value")
  plt.ylabel("Actual value")
  plt.title("Predicted vs. Actual values for\ncustom linear regression model");
  plt.xlim([0, np.max(preds) + 1])
  plt.ylim([0, np.max(y_test) + 1])
  plt.scatter(preds, y_test)
  plt.plot([0, maximum], [0, maximum]);
  plt.show()


main()