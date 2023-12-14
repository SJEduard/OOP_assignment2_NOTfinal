# %%
import numpy as np
from numpy.core.multiarray import array as array
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


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
        newx = np.ones((np.size(x, axis=0), np.size(x, axis=1)+1))
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

class LassoRegression(MultipleLinearRegressor):
    def __init__(self, dimension: int = 0, default_intercept: float = 0,
                penalty: float = 0, alpha: float = 1, gradient: float = 0):
        super().__init__(dimension, default_intercept)
        self._penalty = penalty
        self._alpha = alpha
        self._gradient = gradient
    
    def sign(self, w: np.array) -> np.array:
        for j in w:
            if w[j] > 0:
                w[j] = 1
            elif w[j] == 0:
                w[j] = 0
            else:
                w[j] = -1
        return w
    
    def init_slope(self, dimension) -> np.array:
        strategy = input("Select strategy 1 or 2")
        if strategy == "1":
            '''dimension or dimension +1?????'''
            self._slope = np.random.uniform(low=-1, high=1, size=dimension)
        elif strategy == "2":
            self.slope = np.random.normal(loc=0, scale=1, size=dimension)
        else:
            print("invalid input")

    def train(self, x: np.array, y: np.array) -> None:
        '''TO DO: 
        1. define m, alpha, lambda
        2. make abc for ridge + lasso
        3. how do we choose between the 2 strategies?
        4. log info each iteration in train
        5. add decorators'''
        m=10 #wtvr
        for i in range(m):
            prediction = self.predict(x) #and do what with it.
            print(f"prediction[{i}]: {prediction}")
            n = np.size(x, axis = 0) #check later
            block1 = ((-2)/n)*np.transpose(x)
            block2 = y - np.matmul(x, self._slope) #b included, check later
            block3 = self._penalty*self.sign(self._slope) #also check
            self._gradient = block1*block2 + block3
            self._slope = self._slope - self._alpha*self._gradient

class RidgeRegression(MultipleLinearRegressor):
    def __init__(self, dimension: int = 0, default_intercept: float = 0, penalty: float = 0, alpha: float = 1):
        super().__init__(dimension, default_intercept)
        self.penalty = penalty
        self.alpha = alpha


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

    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression as linreg
    from sklearn.metrics import mean_squared_error

    diabetes = load_diabetes()  # Test dataset
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    # Two categorical features are removed
    data.drop(columns=['sex', 's4'], inplace=True)
    x = data.values
    y = diabetes.target

    model = MultipleLinearRegressor(dimension=np.size(y))
    x = model.preprocessing(x)
    reg = linreg().fit(x, y)

    model.train(x, y)
    model_pred = model.predict(x)
    scikit_pred = reg.predict(x)

    mse_model = mean_squared_error(y, model_pred)
    mse_scikit = mean_squared_error(y, scikit_pred)

    print("MSE ground truth and predictions, this model: ", mse_model)
    print("MSE ground truth and predictions, by Scikit:  ", mse_scikit)

    # The below comments are code to test-run the ModelSaver class.
    # It was switched off, so as not to create files on the user's local pc.

    # model_saver = ModelSaver('json')
    # saveparams = model_saver.save_model_parameters(model, filename='save')
    # model2 = MultipleLinearRegressor(dimension=np.size(y))
    # loadparams = model_saver.load_model_parameters(model2, filename='save')
    # print(model2._slope)

    # showcasing the functionalities of the regression plotter
    # plotter = RegressionPlotter()

    # choose_num_feat = input("Press 1 for one, 2 for two, 3 for all features.")

    # if choose_num_feat == '1':
    #     # we train the model with one feature
    #     model_1 = MultipleLinearRegressor(dimension=np.size(y))
    #     data1 = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    #     x1 = data1[['bp']].values
    #     x1 = model_1.preprocessing(x1)
    #     model_1.train(x1, y)
    #     y_pred1 = model_1.predict(x1)
    #     plotter.choose_plot(x1, y_pred1, model_1._slope, model_1._intercept)
    # elif choose_num_feat == '2':
    #     # we train the model with two features.
    #     model_2 = MultipleLinearRegressor(dimension=np.size(y))
    #     data2 = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    #     x2 = data2[['age', 'bp']].values
    #     x2 = model_2.preprocessing(x2)
    #     model_2.train(x2, y)
    #     y_pred2 = model_2.predict(x2)
    #     plotter.choose_plot(x2, y_pred2, model_2._slope, model_2._intercept)
    # elif choose_num_feat == '3':
    #     plotter.choose_plot(x, model_pred, model._slope, model._intercept)
    # else:
    #     print("Input value not recognized.")

    # #########################################################################

    model.slope = [1,2,3,4]
    model.intercept = 5
    print(model.slope)

# %%
