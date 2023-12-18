# %%
import numpy as np
from numpy.core.multiarray import array as array
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import logging

class MachineLearningModel(ABC):
    @abstractmethod
    def train(self, x: np.array, y: np.array) -> None:
        pass

    def predict(self, x: np.array) -> np.array:
        pass


class MultipleLinearRegressor(MachineLearningModel):
    def __init__(self, dimension: int = 0, default_intercept: float = 0):
        self._slope = np.zeros(dimension)
        self._intercept = default_intercept

    @property
    def intercept(self) -> np.array:
        return self._intercept

    @intercept.getter
    def intercept(self) -> np.array:
        return self._intercept

    @intercept.setter
    def intercept(self, value:float) -> None:
        self._intercept = value
        self._slope[0] = value

    @property
    def slope(self) -> np.array:
        return self._slope

    @slope.getter
    def slope(self) -> np.array:
        return self._slope

    @slope.setter
    def slope(self, params:np.array) -> None:
        self._slope = params
        self._intercept = params[0]

    def preprocessing(self, x: np.array) -> np.array:
        '''
        Adds a leading row of ones to the input data for the scalars,
        to help find the intercept. Returns the modified matrix.

        Args:
            x: x is an array that contains the input values of the model.

        Returns:
            Returns the updated array with a leading column of ones.
        '''
        newx = np.ones((np.size(x, axis = 0), np.size(x, axis = 1)+1))
        newx[:, 1:] = x
        return newx

    def train(self, x: np.array, y: np.array) -> None:
        '''
        Trains the Multiple linear regression model on x and y.
        This function updates the slope- and intercept-attributes of the model.

        Args:
            x: x is a matrix with n rows(number of data points)
            and p columns(number of dimensions)

            y: y is a 1D array that contains the predictions,
            as many as there are data points.

        Returns:

        '''
        xtone = np.matmul(np.transpose(x), x)
        xttwo = np.matmul(np.linalg.inv(xtone), np.transpose(x))

        self._slope = np.matmul(xttwo, y)
        self._intercept = self._slope[0]

    def predict(self, x: np.array) -> np.array:
        '''
        Gives predicted values for the data points in x based
        on previous training or analysis on the linear behaviour
        exhibited in each dimension.

        Args:
            x: x is an array that contains the input values of the model.

        Returns:
            Returns the predicted values of the model.
        '''
        return np.matmul(x[:, 1:], self._slope[1:]) + self._intercept


class Regularization(MultipleLinearRegressor):
    '''This is a parent class for the LassoRegression and RidgeRegression classes.
    Implementing a parent class is ideal in this case, since the two child classes
    have almost identical methods and attributes and differ very little.'''

    def __init__(self, dimension: int = 0, default_intercept: float = 0,
                default_penalty: float = 0, default_alpha: float = 0):
        super().__init__(dimension, default_intercept)
        self.penalty = default_penalty #public so user can customize it according to their dataset
        self._alpha = default_alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.getter
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value:float) -> None:
        self._alpha = value

    def log_info(self, i: int, loss: float, mae: float) -> None:
        '''Logs the iteration number, the Loss function and the Mean Absolute Error attained so far.

        Args: 
        i: iteration number
        loss: the Loss function
        mae: the Mean Absolue Error 

        Returns: None'''

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename="regression.log"
        )

        logging.info(f"Iteration: {i}")
        logging.info(f"Loss function attained: {loss}")
        logging.info(f"Mean Absolute Error attained: {mae}")

    def init_slope(self, dimension: int) -> np.array:
        '''Randomly initializes the slope, according to two strategies that the user can choose between.
        First strategy draws the slope elements from a uniform distribution between -1 and 1.
        Second strategy draws the slope elements from a normal distribution with 0-mean and std 1

        Args: dimension: the size of the slope = number of features in the dataset +1

        Returns: None'''

        strategy = input("Select strategy 1 or 2")
        if strategy == "1":
            print("strat 1 chosen")
            self._slope = np.random.uniform(low=-1, high=1, size=dimension)
        elif strategy == "2":
            print("strat 2 chosen")
            self._slope = np.random.normal(loc=0, scale=1, size=dimension)
        else:
            raise ValueError("Invalid input. Choose 1 or 2")

    def grad_penalty(self) -> np.array:
        '''This function is implemented individually for 
        Lasso and Ridge in their respective classes''' 
        pass

    def compute_pnorm(self) -> float:
        '''In the LassoRegression class, 1-norm is computed and
        In the RidgeRegression class, 2-norm is computed'''
        pass

    def train(self, x: np.array, y: np.array) -> None:
        '''This function trains the model using gradient descent that is computed
        according to the instructions from the assignment. Each iteration, the slope
        array is updated, the prediction of that iteration is computed and the required
        information is logged using the log_info method.

        Args: 
        x: the data set 
        y: the array containing the targets

        Returns: None'''

        n = np.size(x, axis = 0)

        #initializing slope
        self.init_slope(np.size(x, axis=1)) 

        ITERATIONS = 100
        for i in range(ITERATIONS):

            #computing gradient
            block1 = ((-2)/n)*np.transpose(x)
            block2 = y - np.matmul(x, self._slope)
            block3 = self.grad_penalty()
            gradient = np.matmul(block1, block2) + block3

            #updating slope
            self._slope = self._slope - (self._alpha * gradient)

            #computing and logging the required info
            prediction = self.predict(x)
            p_norm = self.compute_pnorm()
            loss = (1/n)*sum((y-prediction)**2) + self.penalty * p_norm
            mae = (1/n)*sum(np.absolute(y-prediction))
            self.log_info(i, loss, mae)


class LassoRegression(Regularization):

    def sign(self, w: np.array) -> np.array:
        '''This function returns an array containing the signs of
        the elements in the parameter array. If the element is positive, the element
        in the new array will be 1, if the element in the parameter array is negative,
        the element in the new array will be -1 and otherwise it will be 0.

        Args: w: the array with the weights for which this function computes the sign array.

        Returns: new_w: an array containing the signs of the elements from w'''

        new_w = np.zeros(np.size(w))
        for j in range(np.size(w)):
            if w[j] > 0:
                new_w[j] = 1
            elif w[j] == 0:
                new_w[j] = 0
            else:
                new_w[j] = -1
        return new_w

    def grad_penalty(self) -> np.array:
        '''This function computes the second summand (block3, as defined in the train function)
        which, for lasso regression, is equal to penalty * sign(slope).

        Args: None

        Returns: block3: product of penalty and the sign of the slope array'''

        block3 = self.penalty*self.sign(self._slope)
        return block3

    def compute_pnorm(self) -> float:
        '''This function computes the 1-norm , which is the sum of the
        absolute values of the elements in the slope.

        Args: None

        Returns: p_norm: the value of the 1-norm '''

        p_norm = sum(np.absolute(self._slope))
        return p_norm


class RidgeRegression(Regularization):

    def grad_penalty(self) -> np.array:
        '''This function computes the second summand (block3, as defined in the train function)
        which, for ridge regression, is equal to 2 * penalty * slope.

        Args: None

        Returns: block3: product of 2*penalty and the slope array'''

        block3 = 2*self.penalty*self._slope
        return block3

    def compute_pnorm(self) -> float:
        '''This function computes the 2-norm , which is the square root of the sum of the
        squared values of the elements in the slope.

        Args: None

        Returns: p_norm: the value of the 2-norm '''

        p_norm = np.sqrt(sum(self._slope **2))
        return p_norm


class ModelSaver:
    def __init__(self, format: str = 'csv'):
        if (format != 'csv') and (format != 'json'):
            raise ValueError("Invalid file format. Choose 'csv' or 'json'.")
        self.format = format

    def save_model_parameters(self, model, filename: str = 'my_output',
                              added_path: str = './') -> None:

        '''
        Reads the parameters of a linear regression model from a trained model
        and writes them into a csv or json file.

        Args:
            model: model contains the parameters of the
            linear regression model.
            filename: The file name of the created file.
            added_path: This contains the path to where
            the created file is to be stored.
        Returns:

        '''
        parameters = model._slope
        parameters[0] = model._intercept
        data = pd.DataFrame(parameters)
        path = f'{added_path}{filename}.{self.format}'
        if self.format == 'csv':
            data.to_csv(path, index=False)
        elif self.format == 'json':
            data.to_json(path, orient='records')

    def load_model_parameters(self, model, filename: str,
                              added_path: str = './') -> None:
        '''
        Reads the parameters of a linear regression model
        from a csv or json file and writes them into the
        slope and intercept of the model.

        Args:
            model: model is the location where the parameters
            of the linear regression model are to be placed.

            filename: The name of the file that is being read.

            added_path: The location where the file that is
            being read is stored.

        Returns:

            '''
        path = f'{added_path}{filename}.{self.format}'
        if self.format == 'csv':
            data = pd.read_csv(path)
        elif self.format == 'json':
            data = pd.read_json(path)
        parameters = data.to_numpy().ravel()

        model._slope = parameters
        model._intercept = parameters[0]

class RegressionPlotter:
    def __init__(self) -> None:
        pass

    def plot_line(self, x: np.array, y: np.array,
                  m: np.array, b: float) -> None:
        '''
        Plots a regression line over a scatterplot of the data points.

        Args:
            x: x contains the horizontal data of the data points
            that are to be scattered.

            y: y contains the targets, meaning the vertical
            data of the data points that are to be scattered.

            m: m contains the slope-parameter of the model,
            for the creation of the regression line.

            b: b contains the intercept of the model.

        Returns:

        '''

        # making x into a 1-d array so we can plot it.
        x = x.flatten()

        plt.scatter(x, y)

        # plot regression line
        plt.plot(x, m[1]*x+b, c='orange')

        plt.xlabel('X Data')
        plt.ylabel('Prediction')
        plt.title('2D Plot: Regression Line')

        plt.show()

    def plot_plane(self, x: np.array, y: np.array,
                   m: np.array, b: float) -> None:
        '''
        Creates a  3D space and plots the regression plane over
        a scatterplot of data points with two attributes.

        Args:
            x: x contains the values in both dimensions of the
            data points that are to be scattered.

            y: y contains the targets, meaning the vertical data
            of the data points that are to be scattered.

            m: m contains the slope-parameters of the model,
            for the creation of the regression plane.

            b: b contains the intercept of the model.

        Returns:

        '''

        x_data = x[:, 0].flatten()
        y_data = x[:, 1].flatten()
        z_data = y

        X, Y = np.meshgrid(x_data, y_data)
        Z = X*m[1]+Y*m[2]+b

        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, alpha=0.3)
        ax.scatter3D(x_data, y_data, z_data, c='black')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Prediction')
        plt.title('3D Plot: Regression Plane')

        plt.show()

    def multidim_plotter(self, x: np.array, y: np.array,
                         m: np.array, b: float) -> None:
        '''
        Creates a series of P plots, where P is the number of dimensions.
        Each plot has one of the features on the horizontal axis
        and the target values on the vertical axis.

        Args:
            x: x contains the horizontal data in each dimension
            for the data points that are to be scattered.

            y: y contains the targets, or the vertical data
            of the data points that are to be scattered.

            m: m contains the slope-parameters of the model,
            for the creation of the regression lines.

            b: b contains the intercept of the model.

        Returns:

        '''
        p_features = len(x[1])

        fig, axs = plt.subplots(p_features, figsize=(6, p_features*6))
        for p in range(p_features):
            x_data = x[:, p].flatten()

            # make a scatter plot
            axs[p].scatter(x_data, y)

            # plot regression line
            axs[p].plot(x_data, m[p+1]*x_data+b,
                        c='purple', label='model parameters regression line')

            # plot a line that fits the data points, obtained by polyfit
            m1, b1 = np.polyfit(x_data, y, 1)
            axs[p].plot(x_data, m1*x_data+b1,
                        c='orange',
                        label='np.polyfit regression line')

            axs[p].set_xlabel(f'Feature {p+1}')
            axs[p].set_ylabel('Prediction')
            axs[p].legend()
            axs[p].set_title(f'Plot {p+1}: Regression Line for Feature {p+1}')

        plt.tight_layout()
        plt.show()

    def choose_plot(self, x: np.array, y: np.array,
                    m: np.array, b: float) -> None:
        '''
        Checks user input to find how many features there are in the dataset,
        and chooses the right plotting function.
        In case that the number of features is exactly two,
        the function lets the user choose if they want to see
        the 3D plot of the regression plane,
        or two separate 2D plots of the two features.

        The arguments of this function are passed to the 'plot_line',
        'plot_plane' or 'multidim_plotter' functions.

        Args:
            x: x contains the horizontal data of the data points
            that are to be scattered.

            y: y contains the vertical data of the data points
            that are to be scattered.

            m: m contains the parameters of the model, for the creation
            of the regression line.

            b: b contains the intercept of the model.

        Returns:

        '''

        # deleting the 1-column
        x = x[:, 1:]
        p_features = len(x[1])

        if p_features == 1:
            self.plot_line(x, y, m, b)

        elif p_features == 2:
            user_choice = input("Press 1 for a 3D plot, 2 for two 2D plots:")
            if user_choice == "1":
                self.plot_plane(x, y, m, b)
            elif user_choice == "2":
                self.multidim_plotter(x, y, m, b)
            else:
                print("There are only 2 features. Choose 1 or 2.")
        else:
            self.multidim_plotter(x, y, m, b)


if __name__ == "__main__": 

    #clearing the log file to add new values
    log_file_path = 'regression.log'
    with open(log_file_path, 'w') as file:
        file.truncate(0)

    x_df = pd.read_csv('data_set')
    y_df = pd.read_csv('targets')

    x = x_df.values
    y = y_df.values
    y = y.flatten()

    reg = Regularization()
    lasso = LassoRegression(dimension=np.size(y), default_alpha=0.01, default_penalty=1)
    ridge = RidgeRegression(dimension=np.size(y), default_alpha=0.001, default_penalty=1)
    x = reg.preprocessing(x)

    choice = input("Press 1 for Lasso regression or 2 for Ridge regression")
    if choice == "1":
        print("You chose Lasso!")
        lasso.train(x, y)
        lasso_pred = lasso.predict(x)
        print(f"lasso slope: {lasso._slope}")
    elif choice == "2":
        print("You chose Ridge!")
        ridge.train(x, y)
        ridge_pred = ridge.predict(x)
        print(f"ridge slope: {ridge._slope}")
    else:
        raise ValueError("Invalid input. Press 1 or 2.")