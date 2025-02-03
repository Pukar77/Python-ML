# ctrl + shift + p -> python interpreter
import numpy as np

class CustomLinearRegression3:
  def __init__(self, alpha=0.01,n_tier=10):
    self.alpha = alpha
    self.n.iter = iter

    def _feature_scaling(self,x):
      

  def train(self,x,y):
    self.n_rec, self.n_features = x.shape

    if isinstance(x,(pd.DataFrame,pd.Series)):
      x = x.to_numpy()

    elif isinstance(x,np.ndarray):
      pass

    else:
      raise Exception("x must be numpy array or pandas Dataframe")

    #initalizing weight, jati ota feature teti ota weight

    self.w = np.random.random(self.n_features)
    self.b = np.random.random()

    for i in range(self.n_iter):
      print(f"Epoch{i+1}")
      y_hat = self.predict(x)

      diff = y_hat - y  #difference for gradient calculation
      loss = self.lose_mse( y,y_hat)


      #gradient calculation 
      grad_b = (2/self.n_rec) * np.sum(diff)
      grad_w = (2/self.n_rec) * np.dot(diff,x)

      self.b = self.b - self.alpha * grad_b
      self.w = self.w - self.alpha * grad_w

      print(f"{grad_b = }, {grad_w = }")
      print(f"{self.b = }, {self.w = }")
      print(f"{loss = }")
      
      

  def predict(self,x):
    return np.dot(x,self.w)
  
  


  def lose_mse(self, y,y_hat):
    return np.mean((y-y_hat)**2)
