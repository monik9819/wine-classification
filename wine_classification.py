# coding: utf-8

# ### Import all libraries

# In[1]:


#importing all the libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# ### Load datasets and Normalize

# In[2]:


df = pd.read_csv('wine.csv')
# print(df)
a = pd.get_dummies(df['Wine'])
df = pd.concat([df,a],axis=1)
X = df.drop([1, 2,3,4,'Wine'], axis = 1)
y = df[[1,2,3,4]].values
X_train, X_test, Y_train,Y_test = train_test_split(X, y, test_size=0.20,)
scaler = StandardScaler()  #it the class for it below are teh function for it
scaler.fit(X_train)        #find the max and mini values of data
X_train = scaler.transform(X_train)   #normalise is the data by mapping it between 0-1 
X_test = scaler.transform(X_test)
# Y_test,test

print("for pretrained model press 1 and to continue to train model with random weights and bais press 2 ")
o=int(input())

# ### Explore dataset

# In[3]:


print(df.head())


# In[5]:


print(df.tail())


# ### Forward Propagation function

# In[6]:


def forward_prop(model,a0):
    
    # Load parameters from model
    W1, b1, W21, b21, W22, b22, W31, b31, W32, b32, W4, b4 = model['W1'], model['b1'], model['W21'], model['b21'], model['W22'], model['b22'], model['W31'], model['b31'], model['W32'], model['b32'], model['W4'], model['b4']
    # Do the first Linear step 
    # Z1 is the input layer x times the dot product of the weights + bias b
    z1 = a0.dot(W1) + b1
    
    # Put it through the first activation function
    a1 = np.tanh(z1)
    
    # Second linear step
    z21 = a1.dot(W21) + b21

    z22 = a1.dot(W22) + b22
    
    # Second activation function
    a21 = np.tanh(z21)

    a22 = np.tanh(z22)

    # third linear step
    z31 = a21.dot(W31) + a22.dot(W31) + b31

    z32 = a21.dot(W32) + a22.dot(W32) + b32

    # third activation function
    a31 = np.tanh(z31)
    a32 = np.tanh(z32)
      
    #fourth linear step
    z4 = a31.dot(W4) + a32.dot(W4) + b4
    
    #For the fourth linear activation function we use the softmax function, either the sigmoid of softmax should be used for the last layer
    a4 = softmax(z4)
    
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z21':z21,'a21':a21,'z22':z22,'a22':a22,'a31':a31,'z31':z31,'a32':a32,'a4':a4,'z4':z4}
    return cache


# ### Softmax Activation function 

# In[7]:


def softmax(z):
    #Calculate exponent term first
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# ### Backpropagation function

# In[8]:


def backward_prop(model,cache,y):    ##

    # Load parameters from model\
    W1, b1, W21, b21, W22, b22, W31, b31, W32, b32, W4, b4 = model['W1'], model['b1'], model['W21'], model['b21'], model['W22'], model['b22'], model['W31'], model['b31'], model['W32'], model['b32'], model['W4'], model['b4']
    
    # Load forward propagation results
    a0, a1, a21, a22, a31, a32, a4 = cache['a0'],cache['a1'],cache['a21'],cache['a22'],cache['a31'],cache['a32'],cache['a4']
    
    # Get number of samples
    m = y.shape[0]
    
    # Calculate loss derivative with respect to output
    dz4 = loss_derivative(y=y,y_hat=a4)

    # Calculate loss derivative with respect to third layer weights
    dW4 = 1/m*(a31.T).dot(dz4) + 1/m*(a32.T).dot(dz4)

    # Calculate loss derivative with respect to third layer bias
    db4 = 1/m*np.sum(dz4, axis=0)
    
  
    
    # Calculate loss derivative with respect to third layer
    dz3 = np.multiply(dz4.dot(W4.T) ,tanh_derivative(a31) + tanh_derivative(a32))
    
   
    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/m*np.dot(a21.T, dz3) + 1/m*np.dot(a22.T, dz3)
       
    # Calculate loss derivative with respect to first layer bias
    db3 = 1/m*np.sum(dz3, axis=0)
    dz2 = np.multiply((dz3.dot(W31.T)+dz3.dot(W32.T)) ,tanh_derivative(a31)+tanh_derivative(a32))
                             
    dW2 = 1/m*np.dot(a1.T, dz2) 
  
    db2 = 1/m*np.sum(dz2, axis=0)

     
    
    dz1 = np.multiply((dz2.dot(W21.T)+dz2.dot(W22.T)),tanh_derivative(a1))
    
    
    dW1 = 1/m*np.dot(a0.T,dz1)
    
    db1 = 1/m*np.sum(dz1,axis=0)
    
    # Store gradients
    grads = {'dW4':dW4, 'db4':db4, 'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads


# ### Loss/Objective/Cost function

# In[9]:


def softmax_loss(y,y_hat):
    # Clipping value
    minval = 0.000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
    return loss


# ### Loss and activation derivative for backpropagation 

# In[10]:


def loss_derivative(y,y_hat):
    return (y_hat-y)

def tanh_derivative(x):
    return (1 - np.power(x, 2))


# ### Randomly initialize all Neural Network parameters

# In[11]:


def initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim):
    # First layer weights
    W1 = 2 * np.random.randn(nn_input_dim, nn_hdim) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim))
    
    # Second layer weights
    W21 = 2 * np.random.randn(nn_hdim, nn_hdim) - 1
    W22 = 2 * np.random.randn(nn_hdim, nn_hdim) - 1
    
    # Second layer bias
    b21 = np.zeros((1, nn_hdim))
    b22 = np.zeros((1, nn_hdim))

    #third layer
    W31 = 2 *  np.random.randn(nn_hdim, nn_hdim) - 1
    b31 = np.zeros((1, nn_hdim))
    W32 = 2 *  np.random.randn(nn_hdim, nn_hdim) - 1
    b32 = np.zeros((1, nn_hdim))
    

    #fourth layer
    W4 = 2 * np.random.rand(nn_hdim, nn_output_dim) - 1
    b4 = np.zeros((1,nn_output_dim))
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W21': W21, 'b21': b21,  'W22': W22, 'b22': b22, 'W31':W31,'b31':b31, 'W32':W32,'b32':b32, 'W4':W4,'b4':b4}
    return model


# ### Update Parameters

# In[12]:


def update_parameters(model,grads,learning_rate):
    # Load parameters
    W1, b1, W21, b21, W22, b22, W31, b31, W32, b32, W4, b4 = model['W1'], model['b1'], model['W21'], model['b21'], model['W22'], model['b22'], model['W31'], model['b31'], model['W32'], model['b32'], model['W4'], model['b4']
   
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W21 -= learning_rate * grads['dW2']
    b21 -= learning_rate * grads['db2']
    W22 -= learning_rate * grads['dW2']
    b22 -= learning_rate * grads['db2']
    W31 -= learning_rate * grads['dW3']
    b31 -= learning_rate * grads['db3']
    W32 -= learning_rate * grads['dW3']
    b32 -= learning_rate * grads['db3']
    W4 -= learning_rate * grads['dW4']
    b4 -= learning_rate * grads['db4']
    
    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W21': W21, 'b21': b21,  'W22': W22, 'b22': b22, 'W31':W31,'b31':b31, 'W32':W32,'b32':b32, 'W4':W4,'b4':b4}
    return model


# ### Predict function

# In[13]:


def predict(model, x):
    # Do forward pass
    c = forward_prop(model,x)
    #get y_hat
    y_hat = np.argmax(c['a4'], axis=1)    
    return y_hat


# ### Train function

# In[15]:
losses=[]

def train(model,X_,y_,learning_rate, iterations, print_loss=False):
    # Gradient descent. Loop over epochs
    for i in range(0, iterations):

        # Forward propagation
        cache = forward_prop(model,X_)

        # Backpropagation
        grads = backward_prop(model,cache,y_)
        
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
    
        # Pring loss & accuracy every 100 iterations
        if print_loss and i % 100 == 0:
            a4 = cache['a4'] 
            print('Loss after iteration',i,':',softmax_loss(y_,a4))   
            y_hat = predict(model,X_)
            y_true = y_.argmax(axis=1)
            print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
            losses.append(accuracy_score(y_pred=y_hat,y_true=y_true)*100)
    pickle.dump(model,open("save.p","wb"))   
    return model


# ### Initialize model parameters and train model on wine dataset  

# In[16]:

if o==2:
    model = initialize_parameters(nn_input_dim=13, nn_hdim= 5, nn_output_dim= 4) 
 
if o==1:
    model = pickle.load(open("save.p","rb"))

model = train(model,X_train,Y_train,learning_rate=0.07,iterations=4500,print_loss=True)
plt.plot(losses)


# ###  Calculate testing accuracy

# In[17]:



test = predict(model,X_test)
test = pd.get_dummies(test)
Y_test = pd.DataFrame(Y_test)
print("Testing accuracy is: ",str(accuracy_score(Y_test, test) * 100)+"%")

plt.show()
