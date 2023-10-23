# How do we calculate the output of a neuron(Logistic Regression):
Just link in a brain of any organism is a collection of neurons which is a building block of a brain in a similar fashon,
a logistic regression is also a building black of a neural network.
#### Here is a diagram of a logistic regression (a single neuron) that is the building block of a neural network.
![](Deep_learning_prerequisite/util_images/logistic_regression.jpg)
Here as you can see in the above diagram there are two circles with X's are multipliers which multiplies the x1 and x2 with the w1 and w2 here (w1 and w2 are the weights). and then there is another circle with nothing in that which is a summer and a non-linear transformer sigmoid(w1x1 + w2x2). So the unique thing about the logistic regression is the circle that comes in front of the output y. It applies logistic function or the sigmoid function.
#### Here is a diagram of a Sigmoid or Logistic function:
![](Deep_learning_prerequisite/util_images/sigmoid_function.png)
![](Deep_learning_prerequisite/util_images/sigmoid_function_formula.png)
Sigmoid Function has a finite limit as X approaches infinity and a finite limit as X approaches minus infinity.Sigmoid function goes from 0 to 1 and it's Y-intercept is 0.5.There are two commonly used Sigmoid functions that are used in AI/ML,
one is hyperbolic tangent or tanh(x) which goes from (-1,1) and it's Y=intercept is 0 and another one is a sigmoid function denoted by a letter called sigma as we have seen above.
#### Here is a diagram of tanh function or hyperbolic tangent function:
![](Deep_learning_prerequisite/util_images/tanh_function.png)
![](Deep_learning_prerequisite/util_images/tanh_function_formula.png)
