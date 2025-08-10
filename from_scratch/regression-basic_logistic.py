import numpy as np
"""
Esta es mi primera red neuronal.
Utiliza un algoritmo ampliamente utilizado en clasificación binaria,
llamado logistic_regression.
Sufre de Overfitting.
Solo tiene una Neurona.

-----NEURONA-----
FUNCIÓN ACTIVACIÓN:
    sigmoid: 1/(1 + np.exp(-A))
    A(X) = w°X + b
    A = np.dot(w.T, X) + b
        w: Pesos
        b: Base/Bias
        X: Vector Inputs

-----ENTRENAMIENTO-----
FUNCIÓN PÉRDIDA:
    L(Y, A) = -[Ylog(A) + (1-Y)log(1-A)]
    L = -(Y*np.log(A) + (1-Y)*np.log(1-A))
        A: Activación Input
        Y: Activación Datos de Entrenamiento

FUNCIÓN COSTO:
    J = np.sum(L)/m
        m: Cantidad de Datos de Entrenamiento
        L: Función de Pérdida

    dJ/dw = X°(A - Y)/m
    dw = np.dot(X, (A - Y).T)/m

    db = np.sum(A - Y)/m

ACTUALIZACIONES DE PARÁMETROS Y DE BASE
    w := w - & * dw
    b := b - & * db
        &: Learning Rate
"""
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def propagate(w, b, X_train, Y_train):
    # Training samples quantity
    m = X_train.shape[1]
    # Linear Function for Activation
    z = np.dot(w.T, X_train) + b
    # Activation Function
    A = sigmoid(z)
    # Loss Function and Cost Function for later optimization for the learning process
    loss = -Y_train*np.log(A) - (1-Y_train)*np.log(1-A)
    cost = 1/m*np.sum(loss)
    # Cost derivatives respect to the weights and bias for gradient descent for optimization
    dw = 1/m*np.dot(X_train, (A-Y_train).T)
    db = 1/m*np.sum(A-Y_train)
    # Saving Gradients in a Dictionary for Returning
    grads = {"dw": dw, "db": db}
    # Return Gradients Dictionary and Cost for optimization
    return grads ,cost

def optimize(w, b, X_train, Y_train, iterations, l_rate):
    # Cost list accumulator for later model analysis
    costs = []
    # Optimizer Loop.
    for i in range(iterations):
        # Frontpropagation. Returns Grads and Cost for Backpropagation
        grads, cost = propagate(w,b,X_train,Y_train)
        dw = grads['dw']
        db = grads['db']
        # Backpropagation. Updating weights and bias.
        w -= l_rate*dw
        b -= l_rate*db
        # Appending cost every 100 iteration
        if i % 100 == 0:
            costs.append(cost)
    # Trained parameters saved in a dictionary for easier processing
    params = {"w": w, "b": b}
    # Final gradients saved in a dictionary for later model analysis.
    grads = {"dw": dw, "db": db}
    # Returns params for predictions, grads and costs for later model analysis.
    return params, grads, costs

def predict(w, b, X_input):
    # Input Samples Quantity
    m = X_input.shape[1]
    # Predictions Accumulator Vector
    Y_prediction = np.zeros((1,m))
    # Reshaping trained weights vector for the Z linear function
    w = w.reshape(X_input.shape[0], 1)
    # Calculating Z linear function for Activation Function
    z = np.dot(w.T, X_input) + b
    # Calculating Activation Function
    A = sigmoid(z)
    # Loop for updating the Y_prediction Vector with predictions
    # over the dataset
    for i in range(A.shape[1]):
        # Activation Determiners.
        # If Activation Function <= 0.5, it is not a cat.
        # Else if Activation Function > 0.5, it is a cat.
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        elif A[0, i] > 0.5:
            Y_prediction[0, i] = 1
    # Return predictions vector Y_prediction
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, iterations, l_rate):
    # Training weights and bias with Training Set
    # START
    w, b = np.zeros((X_train.shape[0], 1)), 0
    params, grads, costs = optimize(w,b,X_train,Y_train,iterations,l_rate)
    w,b = params["w"], params["b"]
    # END

    # Predicting on training set
    Y_prediction_train = predict(w,b,X_train)

    # Predicting on test set
    Y_prediction_test = predict(w,b,X_test)

    # Predictions indicators (accuracy, costs, bias, weights)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : l_rate,
         "num_iterations": iterations}
    return d
