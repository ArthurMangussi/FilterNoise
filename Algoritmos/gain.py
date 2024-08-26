# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from Algoritmos.utils_gain import normalization, renormalization, rounding
from Algoritmos.utils_gain import xavier_init
from Algoritmos.utils_gain import binary_sampler, uniform_sampler, sample_batch_index


class Gain:
  """
  Impute missing values using Generative Adversarial Imputation Nets (GAIN)
  """
  def __init__(self, batch_size:int = 64, hint_rate:float = 0.9, alpha:int = 200, iterations:int = 10000):
    self.batch_size = batch_size
    self.hint_rate = hint_rate
    self.alpha = alpha
    self.iterations = iterations
    self.data_m = None
    self.no, self.dim = None, None
    self.norm_data_x = None

  ## GAIN functions
  # Generator
  def _generator(self,x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3) 
    return G_prob
      
  # Discriminator
  def _discriminator(self,x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
    D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  def fit(self, data_x:np.array):
    
    # Define mask matrix
    self.data_m = 1-np.isnan(data_x)
    
    # Other parameters
    self.no, self.dim = data_x.shape
    
    # Hidden state dimensions
    h_dim = int(self.dim)
    
    # Normalization
    norm_data, self.norm_parameters = normalization(data_x)
    self.norm_data_x = np.nan_to_num(norm_data, 0)
  
    ## GAIN architecture   
    # Input placeholders
    # Data vector
    self.X = tf.placeholder(tf.float32, shape = [None, self.dim])
    # Mask vector 
    self.M = tf.placeholder(tf.float32, shape = [None, self.dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [None, self.dim])
    
    # Discriminator variables
    self.D_W1 = tf.Variable(xavier_init([self.dim*2, h_dim])) # Data + Hint as inputs
    self.D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    self.D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    self.D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    self.D_W3 = tf.Variable(xavier_init([h_dim, self.dim]))
    self.D_b3 = tf.Variable(tf.zeros(shape = [self.dim]))  # Multi-variate outputs
    
    theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]
    
    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    self.G_W1 = tf.Variable(xavier_init([self.dim*2, h_dim]))  
    self.G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    self.G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    self.G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    self.G_W3 = tf.Variable(xavier_init([h_dim, self.dim]))
    self.G_b3 = tf.Variable(tf.zeros(shape = [self.dim]))
    
    theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]
    
    ## GAIN structure
    # Generator
    self.G_sample = self._generator(self.X, self.M)
  
    # Combine with observed data
    Hat_X = self.X * self.M + self.G_sample * (1-self.M)
    
    # Discriminator
    D_prob = self._discriminator(Hat_X, H)
    
    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(self.M * tf.log(D_prob + 1e-8) \
                                  + (1-self.M) * tf.log(1. - D_prob + 1e-8)) 
    
    G_loss_temp = -tf.reduce_mean((1-self.M) * tf.log(D_prob + 1e-8))
    
    MSE_loss = \
    tf.reduce_mean((self.M * self.X - self.M * self.G_sample)**2) / tf.reduce_mean(self.M)
    
    D_loss = D_loss_temp
    G_loss = G_loss_temp + self.alpha * MSE_loss 
    
    ## GAIN solver
    D_solver = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(G_loss, var_list=theta_G)
    
    ## Iterations
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    
    # Start Iterations
    for it in tqdm(range(self.iterations)):    
        
      # Sample batch
      batch_idx = sample_batch_index(self.no, self.batch_size)
      X_mb = self.norm_data_x[batch_idx, :]  
      M_mb = self.data_m[batch_idx, :]  
      # Sample random vectors  
      Z_mb = uniform_sampler(0, 0.01, self.batch_size, self.dim) 
      # Sample hint vectors
      H_mb_temp = binary_sampler(self.hint_rate, self.batch_size, self.dim)
      H_mb = M_mb * H_mb_temp
        
      # Combine random vectors with observed vectors
      X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        
      _, D_loss_curr = self.sess.run([D_solver, D_loss_temp], 
                                feed_dict = {self.M: M_mb, self.X: X_mb, H: H_mb})
      _, G_loss_curr, MSE_loss_curr = \
      self.sess.run([G_solver, G_loss_temp, MSE_loss],
              feed_dict = {self.X: X_mb, self.M: M_mb, H: H_mb})
    
    return self
              
  def transform(self, X):
    ## Return imputed data 

    data_m = 1-np.isnan(X)

    norm_data,_ = normalization(X)
    no, dim = X.shape  

    Z_mb = uniform_sampler(0, 0.01, no, dim) 
    M_mb = data_m
    X_mb = np.nan_to_num(norm_data, 0) 
      
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        
    imputed_data = self.sess.run([self.G_sample], feed_dict = {self.X: X_mb, self.M: M_mb})[0]
    
    imputed_data = data_m * X_mb + (1-data_m) * imputed_data
    
    # Renormalization
    imputed_data = renormalization(imputed_data, self.norm_parameters)  
    
    # Rounding
    imputed_data = rounding(imputed_data, X)  
            
    return imputed_data


