import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            flag = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    flag = False
            if flag:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layer1_dimension = 20
        self.w_1 = nn.Parameter(1, self.layer1_dimension)
        self.b_1 = nn.Parameter(1, self.layer1_dimension)

        self.layer2_dimension = 20
        self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)  # input features, output features
        self.b_2 = nn.Parameter(1, self.layer2_dimension)

        self.layer3_dimension = 1
        self.w_3 = nn.Parameter(self.layer2_dimension, self.layer3_dimension)  # input features, output features
        self.b_3 = nn.Parameter(1, self.layer3_dimension)

        self.batch_size = 0
        self.out_dimension = 1
        self.alpha = -0.01

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        if self.batch_size == 0:
            self.batch_size = x.data.shape[0]

        def linear_t(x, w, b):
            return nn.AddBias(nn.Linear(x, w), b)

        first_layer = nn.ReLU(linear_t(x, self.w_1, self.b_1))
        second_layer = nn.ReLU(linear_t(first_layer, self.w_2, self.b_2))
        return linear_t(second_layer, self.w_3, self.b_3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            flag = 0
            n = 0
            for x, y in dataset.iterate_once(self.batch_size):
                n += 1
                loss = self.get_loss(x, y)
                flag += nn.as_scalar(loss)
                origin = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3]
                grad = nn.gradients(loss, origin)
                for i in range(len(origin)):
                    origin[i].update(grad[i], self.alpha)
            if flag / dataset.x.shape[0] < 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 0
        self.in_dimension = 784
        self.out_dimension = 10
        self.alpha = -0.01
        self.setting = 1

        if self.setting == 1:
            self.layer1_dimension = 100
            self.w_1 = nn.Parameter(self.in_dimension, self.layer1_dimension)  # input features, output features
            self.b_1 = nn.Parameter(1, self.layer1_dimension)  # 1 * output features

            self.layer2_dimension = 70
            self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)  # input features, output features
            self.b_2 = nn.Parameter(1, self.layer2_dimension)

            self.layer3_dimension = self.out_dimension
            self.w_3 = nn.Parameter(self.layer2_dimension, self.layer3_dimension)  # input features, output features
            self.b_3 = nn.Parameter(1, self.layer3_dimension)

        else:
            self.layer1_dimension = 100
            self.w_1 = nn.Parameter(self.in_dimension, self.layer1_dimension)  # input features, output features
            self.b_1 = nn.Parameter(1, self.layer1_dimension)  # 1 * output features

            self.layer2_dimension = self.out_dimension
            self.w_2 = nn.Parameter(self.layer1_dimension, self.layer2_dimension)  # input features, output features
            self.b_2 = nn.Parameter(1, self.layer2_dimension)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        if self.batch_size == 0:
            self.batch_size = x.data.shape[0]

        def linear_t(x, w, b):
            return nn.AddBias(nn.Linear(x, w), b)

        if self.setting == 1:
            first_layer = nn.ReLU(linear_t(x, self.w_1, self.b_1))
            second_layer = nn.ReLU(linear_t(first_layer, self.w_2, self.b_2))
            return linear_t(second_layer, self.w_3, self.b_3)
        else:
            first_layer = nn.ReLU(linear_t(x, self.w_1, self.b_1))
            return linear_t(first_layer, self.w_2, self.b_2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            n = 0
            for x, y in dataset.iterate_once(self.batch_size):
                n += 1
                loss = self.get_loss(x, y)
                if self.setting == 1:
                    origin = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3]
                else:
                    origin = [self.w_1, self.b_1, self.w_2, self.b_2]
                grad = nn.gradients(loss, origin)
                for i in range(len(origin)):
                    origin[i].update(grad[i], self.alpha)
            if dataset.get_validation_accuracy() > 0.96:
                self.alpha = -0.003
            if dataset.get_validation_accuracy() > 0.972:
                break

class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = None
        self.numTrainingGames = None
        self.batch_size = None

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"