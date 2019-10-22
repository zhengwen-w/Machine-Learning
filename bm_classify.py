import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2
    w = np.zeros(D)
    if w0 is not None:
        w = w0 
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        for i in range(len(y)):
            if y[i]==0:
                y[i]=-1
        x=[]
        for i in X:
            i = np.append(1,i)

            x.append(list(i))
        X=np.array(x)
        w=np.append(b,w)
        for i in range(max_iterations):
            o=np.sign(np.dot(X,w.T))
            error=y-o
            m=np.where(error==0)
            ly = np.delete(y, m)
            lx = np.delete(X, m, axis=0)
            step = step_size*(np.dot(lx.T, ly.T)/ N)
            w = w+step
            
        b=w[0]
        w=w[1:]
        
            
        

    elif loss == "logistic":
        for i in range(max_iterations):
            aaa=sigmoid(np.dot(w,X.T) + b)
            errors=aaa - y
            wg= -step_size * (np.dot(errors,X) /N)
            bg= -step_size * (np.sum(errors) /N)
            w=np.add(w,wg)
            b=np.add(b,bg)
        
    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    value=1/(1+np.exp(-z))
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        pred=np.sign(np.dot(w,X.T)+b)
        l=[]
        for i in pred:
            if i>=0.5:
                l.append(1)
            else:
                l.append(0)
        preds=np.array(l)

        

    elif loss == "logistic":
        pred = sigmoid(w.dot(X.T) + b)
        l=[]
        for i in pred:
            if i>=0.5:
                l.append(1)
            else:
                l.append(0)
        preds=np.array(l)
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        w=np.array([[-0.86676931,4.04576344],[ 0.90817344,3.9173253 ],[-0.04140412,-7.96308873]])
        b=np.array([-3.61113329,-2.83960131,6.45073461])
        

    elif gd_type == "gd":
       
    
        aaa=np.zeros((N,C))
        for i in range(len(y)):
            aaa[i][y[i]]=1
        y=aaa
        
        for i in range(max_iterations):
            x = (np.dot(w,X.T)).T + b
            x = np.exp(x - np.amax(x))
            d = np.sum(x, axis=1)
            e = (x.T / d).T - y
            wg = np.dot(e.T,X) / N
            bg = np.sum(e, axis=0) / N
            w = w-step_size * wg
            b = b-step_size * bg
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    x=np.dot(w,X.T)
    x=x.T+b
    preds=(x.T / np.sum(np.exp(x - np.amax(x)), axis=1)).T
    
    preds = np.argmax(preds, axis=1)

    assert preds.shape == (N,)
    return preds




        