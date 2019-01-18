import numpy as np

from sklearn import datasets
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression

class LogisticRegression:
    """
            Purpose: To estimate Logistic regression parameters in Python.
            Inputs:
                alpha           : Is the optimisation learning rate.
                maxIterations   : Maximum number of iterations for optimisation routine..
                fitIntercept    : Include the intercept in the model fit.
                verbose         : Display program information.
                optimisation    : The optimisation routine to use. Options are:
                                            gradientAscent
                                            newton
    """
    def __init__(self, alpha=0.01, maxIterations=100000, fitIntercept=True, verbose=False,optimisation="gradientAscent"):
        self.alpha = alpha
        self.maxIterations = maxIterations  # Maximum number of times to run the optimisation.
        self.numIterations = 0;             # Record the number of iterations performed.
        self.hasConverged = False;          # This variable is used to terminate the iterations searching for optimum parameters.
        self.fitIntercept = fitIntercept
        self.verbose = verbose
        self.costHistory = [];
        self.tolerance = tol=1e-7; # convergence tolerance;
        self.theta=[];
        self.optimisation = optimisation;

    def __add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))


    def __cost(self, X, y, theta):
        # Purpose: Logistic regression log cost function.
        z = np.dot(X, theta)
        p = LogisticRegression.__sigmoid(z)
        return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()

    def gradientAscent(X,y,theta,alpha):
        z = np.dot(X, theta)
        p = LogisticRegression.__sigmoid(z)
        gradient = np.dot(X.T, (p - y)) / y.size;  # 1st derivative of log likelihood wrt parameters.
        theta -= alpha * gradient  # Update the parameters.

        return theta;

    def newton(X,y,theta,useRegulisation=False,regulisationParameter=0):
        """ Newton optimisation method."""
        z = np.dot(X, theta)
        p = LogisticRegression.__sigmoid(z)
        W = np.diag(p * (1 - p))
        hessian = X.T.dot(W).dot(X);

        gradient = np.dot(X.T, (y-p));  # 1st derivative of log likelihood wrt parameters.

        try:
            if useRegulisation:
                step = np.dot(np.linalg.inv(hessian + regulisationParameter * np.eye(theta)), grad)
            else:
                step = np.dot(np.linalg.inv(hessian), gradient)
        except np.linalg.LinAlgError:
            step=0;

        ## update the weights
        theta = theta + step


        return theta;

    def fit(self, X, y):
        if self.fitIntercept:
            X = LogisticRegression.__add_intercept(X=X)


        self.theta = np.zeros(X.shape[1]);  # Initialise weights.
        alpha=self.alpha;

        cost = self.__cost(X, y, self.theta);  # Calculate the cost.
        self.costHistory.append(cost);  # Record the intitial cost for plotting.


        iterCount=0;

        while not self.hasConverged:
            # Perform the optimisation many times to reduce the cost by improving the parameters.
            iterCount+=1;                       # Counter for the number of optimisation iterations.

            # Depending on the optimisation approach. Calculate the coeffient update step differently.
            if self.optimisation=="gradientAscent":
                theta = LogisticRegression.gradientAscent(X,y,self.theta,self.alpha)
                iterDisplayVerbose = 10000;
            elif self.optimisation=="newton":
                theta = LogisticRegression.newton(X, y, self.theta)
                iterDisplayVerbose = 1;             # The number of iteration steps is significantly less than Gradient Ascent.
            else:
                assert "unknown optimisation routine."
                return;

            cost = self.__cost(X,y, theta);     # Calculate the cost.


            if iterCount>1:
            # Only check to terminate optimisation after performing the second optimisation calculation.
                hasConverged = self.__checkConvergence(self.costHistory[-1], cost, self.tolerance,iterCount);  # Check if should terminate iteration updates as convergence tolerance has been reached.

                if hasConverged.hasConverged==True:
                    print("Iteration #:  {:>7,.0f}.  Cost: {:>+7.4f}.".format(iterCount, cost));
                    print("Finished because {}. Using {} optimisation method.".format(hasConverged.reason, self.optimisation));
                    self.numIterations = iterCount;
                    self.hasConverged == True;

                if (self.verbose == True and iterCount % iterDisplayVerbose == 0) and  hasConverged.hasConverged==False:
                    # Print out the log output.
                    print("Iteration #:  {:>7,.0f}.  Cost: {:>+7.4f}".format(iterCount, cost));


            if iterCount < 2:

                if (self.verbose == True and iterCount % iterDisplayVerbose == 0):
                    # Print out the log output.
                    print("Iteration #:  {:>7,.0f}.  Cost: {:>+7.4f}".format(iterCount, cost));

                self.theta = theta;
                self.costHistory.append(cost);  # Record the cost for plotting.
            else:
                if not (((self.optimisation == "newton") and (cost > self.costHistory[-1])) or (np.isnan(cost))) or iterCount<1:
                    # The Newton method on the last step can give coefficients well off and a worse cost as close to the singularity.
                    # Because of this. Not recording the last theta and cost found in this case and using the previous one.
                    self.theta = theta;
                    self.costHistory.append(cost);  # Record the cost for plotting.


    def __checkConvergence(self,previousCost, cost, tolerance, iterCount):
        ''' Purpose: Checks if coefficients have converged.
            Returns True if they have converged, False otherwise.'''
        costChange = np.abs(previousCost - cost)

        self.reason="";
        self.hasConverged=False;
        if (np.any(costChange < tolerance)):
            self.reason = "cost function tolerance reached";
            self.hasConverged = True;

        # If havn't reached thresholds, perform more iterations (keep training).
        if (iterCount > self.maxIterations):
            self.reason="maximum iterations reached"
            self.hasConverged = True;


        if (self.optimisation == "newton"):
            if (np.isnan(cost)):
                # The Newton method on the last step can give coefficients well off and a worse cost as close to the singularity.
                # Because of this. Not recording the last theta and cost found in this case and using the previous one.
                self.reason="singular Hessian"
                self.hasConverged = True;
            elif (cost > previousCost):
                # The Newton method on the last step can give coefficients well off and a worse cost as close to the singularity.
                # Because of this. Not recording the last theta and cost found in this case and using the previous one.
                self.reason="cost function worsoning as close to solution"
                self.hasConverged = True;



        return self


    def predict_prob(X,theta,fitIntercept=True):
        if fitIntercept:
            X = LogisticRegression.__add_intercept(X=X)

        return LogisticRegression.__sigmoid(np.dot(X, theta))

    def predict(self, X):
        return LogisticRegression.predict_prob((X).round(),self.theta,self.fitIntercept)


    def formattedOutput(objLogisticRegression):
        """
        Purpose: To produce readable output summary of the results.
            Input:
                LogisticRegression  : A class instance containing the fitted information.
        """
        optimisationMethod=objLogisticRegression.optimisation;
        theta=objLogisticRegression.theta;
        inititialCost=objLogisticRegression.costHistory[0];
        finalCost=objLogisticRegression.costHistory[-1];
        numIterations=objLogisticRegression.numIterations;


        dash = '=' * 80; #chr(10000)*50
        print(dash)
        print("LOGISTIC REGRESSION USING {0} TERRMINATION RESULTS".format(optimisationMethod.upper()))
        print(dash)
        print("Initial Weights were:    {:>12.1f}, {:>2.1f}, {:>2.1f}.".format(0, 0, 0))
        print("   With initial cost:    {:>+12.6f}.".format(inititialCost))
        print("        # Iterations:    {:>+12,.0f}.    ".format(numIterations))
        print("       Final weights:    theta0:{:>+0.2f}, theta1:{:>+3.2f}, theta02:{:>+3.3f}.".format(
            theta[0], theta[1], theta[2]))#print("       Final weights:    \u03F4\u2080:{:>+0.2f}, \u03F4\u2081:{:>+3.2f}, \u03F4\u2082:{:>+3.3f}.".format(theta[0], theta[1], theta[2]))
        print("          Final cost:    {:>+12.6f}.".format(finalCost))
        print(dash)






def run():

    # Load Data:
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1


    # Run the model:
    objLogisticRegression = LogisticRegression(alpha=0.1, maxIterations=100000,fitIntercept=True, verbose=True,optimisation="newton");   # Initialise the regression.
    objLogisticRegression.fit(X, y);                                 # Fit the regression.

    # Show the output:
    LogisticRegression.formattedOutput(objLogisticRegression=objLogisticRegression);        # Show the formatted results.



    # Run the model:
    objLogisticRegression = LogisticRegression(alpha=0.1, maxIterations=100000,fitIntercept=True, verbose=True,optimisation="gradientAscent");   # Initialise the regression.
    objLogisticRegression.fit(X, y);                                 # Fit the regression.

    # Show the output:
    LogisticRegression.formattedOutput(objLogisticRegression=objLogisticRegression);        # Show the formatted results.


    # sklearn's Logistic Regression.
    model = sklearnLogisticRegression(C=1e8).fit(X, y)
    dash = '=' * 80;  # '=' * 80;
    print(dash)
    print("LOGISTIC REGRESSION USING SKLEARN TERMINATION RESULTS")
    print("Final weights:    theta0:{:>+0.2f}, theta1:{:>+0.2f}, theta2:{:>+0.2f}.".format(model.intercept_[0], model.coef_[0][0], model.coef_[0][1]))
    print(dash)



    print("Finished")





if __name__ == '__main__':
    run()