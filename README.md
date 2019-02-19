<h2>MachineLearning-LogisticRegressionWithGradientAscentOrNewton</h2>
<h3>Description:</h3>
<ul style="list-style-type:disc">
	<li>Python script to estimate coefficients for Logistic regression using either Gradient Ascent or Newton-Raphson optimisaiton algorithm. Further can choose none/one/both of Ridge and LASSO regularisation. </li>
	<li>Logistic regression implemented from scratch.</li>
	<li>Using the Iris dataset available in sklearn, which contains characteristics of 3 types of Iris plant and is a common dataset when experimenting with data analysis. To learn more about the dataset, click <a href="https://archive.ics.uci.edu/ml/datasets/iris">here</a>.</li>
</ul>


<p float="left">
  <img src="images/logisticRegressionBoundaryFancy.png" width="400" alt="A fancy view of Logistic regression boundary."/>
  <img src="images/logisticRegressionBoundarySimple.png" width="400"alt="A simple view of Logistic regression boundary."/>
</p>
<p float="left">
  <img src="images/logisticRegressionGradientAscentCost.png" width="400" alt="Cost of Gradient Ascent algorithm improvement through epochs."/>
  <img src="images/logisticRegressionNewtonCost.png" width="400"alt="Cost of Newton-Raphson algorithm improvement through epochs."/>
</p>
 

<h3>Model:</h3>
Estimate Logistic equation
<p align="center"><img src="svgs/77f007d76a70b0d25e05fe3e7c470caa.svg" align=middle width=107.73545145pt height=34.3600389pt/></p>

Where <img src="svgs/282f38ecf82d8d7b9d2813044262d5f3.svg" align=middle width=9.347490899999991pt height=22.831056599999986pt/> is given by

<p align="center"><img src="svgs/485220363cbc49355345c0869b2efbdc.svg" align=middle width=608.6145483pt height=18.7598829pt/></p>

And estimates are trained using optimisation of the conditional maximum Likelihood (cost) function

<p align="center"><img src="svgs/ec97e55044e066eda74b67178cbac5b1.svg" align=middle width=429.7248631499999pt height=47.60747145pt/></p>

using either Gradient Ascent or Newton-Raphson methods.
 
<h4>Gradient Ascent</h4> 
The parameter iterative updates are calculated as
<p align="center"><img src="svgs/248c208805e4a629ea7ce79de37903bf.svg" align=middle width=241.29400679999998pt height=92.10448995pt/></p>


<h4>Newton-Raphson</h4> 

The parameter iterative updates are calculated as

<p align="center"><img src="svgs/e784cf3c8a6a969c61573defc296325a.svg" align=middle width=292.36726035pt height=153.69754785pt/></p>

Convergence is reached when either the tolerance level on the cost function has been reached
<p align="center"><img src="svgs/dd837991786bbfa0bd84fa96a9ef45d9.svg" align=middle width=204.557133pt height=16.438356pt/></p>
 or the full Hessian is no longer invertible or the maximum number of iterations has been exceeded.
 
 
<h4>Regularisation</h4> 
<p>
None, either or both LASSO (least absolute shrinkage and selection operator) Regression (L1) or Ridge Regression (L2) are implemented using the mixing parameter <img src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg"  style="padding-top: 5px;vertical-align:middle;width:12pt; height:50pt"/>. Where Ridge <img src="svgs/8721ef437877ca67d513db00210345fe.svg" style="padding-top: 5px; vertical-align:middle;width:47pt; height:20.6pt"/> and Lasso <img src="svgs/cbf9f67dff9c696b64cb0671c65cac33.svg"  style="padding-top: 5px; vertical-align:middle; width:44.51pt; height:20.65pt"/>.
</p>

<p align="center"><img src="svgs/LogisticConditionalLikelihoodWithRidgeAndLassoRegression.png" align=middle/></p>
 
<h3>Decision Boundary</H3>
The linear decision boundary shown in the figures results from setting the target variable to zero and rearranging equation (1). ie.

<p align="center"><img src="svgs/c70f07f91936ea10cae16ace450a0984.svg" align=middle width=260.2377294pt height=79.1309904pt/></p>



 
<h3>How to use</h3>
<pre>
python logisticRegression.py
</pre>
		
		
<h3>Expected Output</h3>
<pre>
 Iteration #:        1.  Cost: +0.2190
 Iteration #:        2.  Cost: +0.1058
 Iteration #:        3.  Cost: +0.0554
 Iteration #:        4.  Cost: +0.0301
 Iteration #:        5.  Cost: +0.0166
 Iteration #:        6.  Cost: +0.0091
 Iteration #:        7.  Cost:    +nan.
 Finished because singular Hessian. Using newton optimisation method.
 ================================================================================
 LOGISTIC REGRESSION USING NEWTON TERRMINATION RESULTS
 ================================================================================
 Initial Weights were:             0.0, 0.0, 0.0.
 With initial cost:       +0.693147.
 # Iterations:              +7.    
 Final weights:    theta0:-25.51, theta1:+11.25, theta02:-11.283.
 Final cost:       +0.009061.
 ================================================================================

	Iteration #:   10,000.  Cost: +0.0343
	Iteration #:   20,000.  Cost: +0.0288
	Iteration #:   30,000.  Cost: +0.0257
	Iteration #:   40,000.  Cost: +0.0234
	Iteration #:   50,000.  Cost: +0.0215
	Iteration #:   60,000.  Cost: +0.0199
	Iteration #:   70,000.  Cost: +0.0185
	Iteration #:   80,000.  Cost: +0.0173
	Iteration #:   88,543.  Cost: +0.0164.
 	Finished because cost function tolerance reached. Using gradientAscent optimisation method.
	================================================================================
	LOGISTIC REGRESSION USING GRADIENTASCENT TERRMINATION RESULTS
	================================================================================
	Initial Weights were:             0.0, 0.0, 0.0.
   	With initial cost:       +0.693147.
	# Iterations:         +88,543.    
	Final weights:    theta0:-13.42, theta1:+9.09, theta02:-11.539.
	Final cost:       +0.016394.
	================================================================================

	================================================================================
	LOGISTIC REGRESSION USING SKLEARN TERMINATION RESULTS
	Final weights:    theta0:-80.54, theta1:+31.59, theta2:-28.30.
	================================================================================

	Finished
</pre>

<h3>Highlights</h3>
<ul style="list-style-type:disc">
	<li>Newton-Raphson optimisation clearly locates coefficients in far less iteration steps than Gradient Ascent.</li>
	<li>Logistic regression is a powerful classification tool in machine learning.<li>


<h3>Requirements</h3>
 <p><a href="https://www.python.org/">Python (>2.7)</a>, <a href="http://www.numpy.org/">Numpy</a> and <a href="https://scikit-learn.org/stable/">Scikit-Learn</a>.</p>
 
 
 
 
