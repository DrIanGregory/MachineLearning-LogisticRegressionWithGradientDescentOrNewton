# Import libraries:
import numpy as np
import pandas as pd
from random import random, seed


def rmse(Y, Ypred):
    # Purpose: Root Mean Square Error (RMSE)
    # Useful in Model Evaluation.
    rmse = np.sqrt(sum((Y - Ypred) ** 2) / len(Y))
    return rmse

def r2Score(Y, Y_pred):
    # Purpose: Calculate R2 for goodness of fit.
    # Useful in Model Evaluation.
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2





def costFunction(X, Y, W):
    # Purpose: Calculate the cost function for multi-linear regression.
    # Inputs:
    #       X       :  Data inputs.
    #       Y       :  Data inputs.
    #       W       :  Weights.
    N = len(Y)
    C = np.sum((X.dot(W) - Y) ** 2)/(2 * N);
    return C

def gradientDescent(X, Y, W, alpha, maxNumIterations=10000):
    # Purpose: Perform gradient decent algorithm given dataset.
    # Inputs:
    #       X                   :       X data      :       Pandas Dataframe of 1 or more columns..
    #       Y                   :       Y data.     :       Pandas Dataframe of 1 column.
    #       alpha               :       Optimisation learning rate.
    #       maxNumIterations    :       Optimisation maximum number of iterations.

    N = len(Y)
    costHistory=[];
    wHistory=[];
    iteration = 0;
    while iteration <maxNumIterations:
        # Hypothesis Values
        h = X.dot(W)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / N
        # Updating slope values using Gradient
        W = W - alpha * gradient
        # New Cost Value
        cost = costFunction(X, Y, W);
        costHistory.append(cost);     # Record the cost (not needed, nice to have for performance analysis).
        iteration = iteration+1;

        wHistory.append(W); # Record the weights at each iteration.
    return W, costHistory,wHistory






def showResults(X,Y,W,newW,costHistory,maxNumIterations,wHistory):
    # Purpose: To display the results.

    inital_cost = costFunction(X, Y, W);
    Y_pred = X.dot(newW)

    dash = '=' * 80;
    print(dash)
    print("MULTI LINEAR REGRESSION USING GRADIENT DESCENT TERMINATION RESULTS")
    print(dash)
    print("Initial Weights were:    {:>12.1f}, {:>2.1f}, {:>2.1f}.".format(W[0],W[1],W[2]))
    print("   With initial cost:    {:>12,.1f}.".format(inital_cost))
    print("        # Iterations:    {:>12,.0f}.    ".format(maxNumIterations))
    print("       Final weights:    w0:{:>+0.2f}, w1:{:>+3.2f}, w2:{:>+3.3f}.".format(newW[0], newW[1], newW[2]))
    print("          Final cost:    {:>+12.1f}.".format(costHistory[-1]))
    print("                RMSE:    {:>+12.1f}, R-Squared: {:>+12.1f}".format(rmse(Y, Y_pred),r2Score(Y, Y_pred)))
    print(dash)


def programBody(data,alpha,maxNumIterations):
    # Purpose: To perform the calculation and show the results.

    # Initialise variables:
        # Initial Coefficients:
    numXColumns = data.shape[1]-1;
    W = (numXColumns+1)*[0];         # Intercepts initialised to zero for the number of features supplied.
    x0 = np.ones(data.shape[0]);

    X = np.column_stack((x0,data.iloc[:, 1:(numXColumns + 1)].values)); # Supplied X data.
    Y = np.array(data.iloc[:,0]);            # Supplied Y data.

    newW, costHistory,wHistory = gradientDescent(X, Y, W, alpha,maxNumIterations)

    showResults(X, Y, W, newW, costHistory, maxNumIterations,wHistory);

    return


def run():
    # Purpose: A starting point for the program. Take any user inputs.

    # User Inputs:
        # Optimisation Parameters :
    alpha = 0.0001;              # Optimisation learning rate.
    maxNumIterations = 2500000;  # Maximum number of optimisation iterations.


    # Generate Data:
    np.random.seed(1234)# Seed random number generator.
    numDataPoints=500;
    means = [70, 70]
    stds = [9,9]
    corr = 0.8  # correlation
    covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],[stds[0] * stds[1] * corr, stds[1] ** 2]]
    data1 = np.random.multivariate_normal(means, covs, numDataPoints).T

    JobProbabilities = (data1[0] + data1[1])/2.5 + np.random.normal(loc=25, scale=4, size=numDataPoints);

    data = np.vstack((JobProbabilities, data1));
    data = pd.DataFrame({"JobPotential":data[0],"AI":data[1],"MachineLearning":data[2]});

    # Run the program
    programBody(data, alpha, maxNumIterations);
    print("Finished");


if __name__ == '__main__':
    run()
