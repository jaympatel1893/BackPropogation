#Author Jay And Sunny

import math
import random
import string

class NN:
  def __init__(self, NI, NH, NO):
    # number of nodes in layers
    self.ni = NI + 1 # +1 for bias
    self.nh = NH
    self.no = NO
    
    # initialize node-activations
    self.ai, self.ah, self.ao = [],[], []
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no

    # create node weight matrices
    self.wi = makeMatrix (self.ni, self.nh)
    self.wo = makeMatrix (self.nh, self.no)
    # initialize node weights to random vals
    randomizeMatrix ( self.wi, -0.2, 0.2 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )
    # create last change in weights matrices for momentum
    self.ci = makeMatrix (self.ni, self.nh)
    self.co = makeMatrix (self.nh, self.no)
    
  def runNN (self, inputs):
    if len(inputs) != self.ni-1:
      print 'incorrect number of inputs'
    
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
      
    for j in range(self.nh):
      sum = 0.0
      for i in range(self.ni):
        sum +=( self.ai[i] * self.wi[i][j] )
      self.ah[j] = sigmoid (sum)
    
    for k in range(self.no):
      sum = 0.0
      for j in range(self.nh):        
        sum +=( self.ah[j] * self.wo[j][k] )
      self.ao[k] = sigmoid (sum)
      
    return self.ao
      
      
  
  def backPropagate (self, targets, N, M):
    
    output_deltas = [0.0] * self.no
    for k in range(self.no):
      error = targets[k] - self.ao[k]
      output_deltas[k] =  error * dsigmoid(self.ao[k]) 
   
    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change

    # calc hidden deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * dsigmoid(self.ah[j])
    
    #update input weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        #print 'activation',self.ai[i],'synapse',i,j,'change',change
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change
        
    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    error = 0.0
    for k in range(len(targets)):
      error = 0.5 * (targets[k]-self.ao[k])**2
    return error
        
        
  def weights(self):
    print 'Input weights:'
    for i in range(self.ni):
      print self.wi[i]
    print
    print 'Output weights:'
    for j in range(self.nh):
      print self.wo[j]
    print ''
  
  def test(self, patterns):
    for p in patterns:
      inputs = p[0]
      print 'Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1]
  
  def train (self, patterns, max_iterations = 1000, N=0.5, M=0.1):
    for i in range(max_iterations):
      for p in patterns:
        inputs = p[0]
        targets = p[1]
        self.runNN(inputs)
        error = self.backPropagate(targets, N, M)
      if i % 50 == 0:
        print 'Combined error', error
    self.test(patterns)
    

def sigmoid (x):
  return math.tanh(x)
  
def dsigmoid (y):
  return 1 - y**2

def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)

def main ():
  pat = [
      [[-0.8,-0.1,-0.5,0.4,0.6], [1,-1,-1]],
      [[0.8,0.1,-0.5,0.4,-0.6], [-1,1,-1]],
      [[0.8,0.1,-0.5,-0.4,0.6], [-1,-1,1]],
      [[-0.9,-0.2,-0.6,0.6,0.7], [1,-1,-1]]
  ]

  # 5 input layers, 10 Hidden layers, 3 output layers

  myNN = NN ( 5, 10, 3)
  myNN.train(pat)
  
  





if __name__ == "__main__":
    main()
