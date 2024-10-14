import copy
import numpy as np
from sklearn.linear_model import LogisticRegression as LR  # type: ignore
from sklearn.metrics.pairwise import pairwise_distances
from tqdm.auto import tqdm  # type: ignore

def GenXY(seen_items, Y, X, Z):
    """
    Generates combined feature and label matrices.

    Parameters:
    seen_items (array-like): Matrix indicating which items have been seen.
    Y (array-like): Target values matrix.
    X (array-like): Feature matrix for the first set of features.
    Z (array-like): Feature matrix for the second set of features.

    Returns:
    tuple: Combined feature matrix and corresponding labels.
    """
    Y_seen = -np.ones(Y.shape)
    Y_seen[seen_items] = Y[seen_items]

    W_x = []
    W_z = []

    labels = []
    for i in range(seen_items.shape[0]):
        for j in range(seen_items.shape[1]):
            if seen_items[i, j] == True:
                W_x.append(X[i])
                W_z.append(Z[j])
                labels.append(Y_seen[i, j])

    W = np.hstack((np.vstack(W_x), np.vstack(W_z)))
    labels = np.array(labels)
    return W, labels


def sigmoid(x):
    """
    Applies the sigmoid function to the input.

    Parameters:
    x (array-like): The input data.

    Returns:
    array-like: The output of the sigmoid function.
    """
    x_clipped = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x_clipped))
    
class LogisticRegression:
    """
    Logistic regression model.

    Attributes:
        reg (float): The regularization parameter for the logistic regression model. This is equivalent to the prior Gaussian covariance scaling in the Bayesian setup with gaussian, ie, prior cov = reg*identity.
    """

    def __init__(self, reg=1e4):
        """
        Initializes the logistic regression model with a regularization parameter.

        Parameters:
            reg (float): Regularization parameter (default is 100).
        """
        self.reg = reg

    def fit(self, X, y):
        """
        Fits the logistic regression model to the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target vector of 0s and 1s.
        """
        # This block of code is just a trick to run the Scikit-Learn implementation for logistic regression
        if np.var(y) == 0:
            y_copy = copy.deepcopy(y)
            local_state = np.random.RandomState(0)
            ind = local_state.choice(len(y_copy))
            y_copy[ind] = 1 - np.median(y_copy)
        else:
            y_copy = copy.deepcopy(y)

        # Fitting the model
        logreg = LR(C=self.reg, random_state=0, solver="liblinear", fit_intercept=False).fit(X, y_copy)
        self.mu = logreg.coef_.squeeze()


class ExtendedRaschModel:
    """
    An extended Rasch model incorporating covariates for both formats and examples.

    Attributes:
        seen_examples (array-like): Boolean array indicating seen examples.
        Y (array-like): Target matrix of 0s and 1s.
        X (array-like): Covariates for formats.
        Z (array-like): Covariates for examples.
        x_dim (int): Dimension of X.
        z_dim (int): Dimension of Z.
        n_formats (int): Number of formats.
        n_examples (int): Number of examples.
        rasch_model (LogisticRegression): The fitted logistic regression model.
        gammas (array-like): Coefficients for the format covariates.
        thetas (array-like): Format parameters.
        psi (array-like): Coefficients for the example covariates.
        betas (array-like): Example parameters.
        logits (array-like): Logits of the fitted model.
    """

    def __init__(self):
        """
        Initializes the extended Rasch model.
        """
        pass

    def fit(self, seen_examples, Y, X=None, Z=None):
        """
        Fits the extended Rasch model to the data.

        Parameters:
            seen_examples (array-like): Boolean array indicating seen examples.
            Y (array-like): Target matrix.
            X (array-like): Covariates for formats (default is identity matrix).
            Z (array-like): Covariates for examples (default is identity matrix).
        """
        self.seen_examples = seen_examples
        self.Y = Y

        # X (formats covariates)
        if type(X) != np.ndarray:
            self.X = np.eye(Y.shape[0])
        else:
            self.X = X
            check_multicolinearity(X)
        self.x_dim = self.X.shape[1]

        # Z (examples covariates)
        if type(Z) != np.ndarray:
            self.Z = np.eye(Y.shape[1])
        else:
            self.Z = Z
            check_multicolinearity(Z)
        self.z_dim = self.Z.shape[1]

        # Formatting the data
        self.n_formats, self.n_examples = seen_examples.shape
        features, labels = GenXY(seen_examples, Y, self.X, self.Z)

        if type(X) != np.ndarray and type(Z) != np.ndarray:  # basic Rasch model (no need to include intercept)
            features = features[:, :-1]
        elif (
            type(X) != np.ndarray or type(Z) != np.ndarray
        ):  # just one set of covariates (no need to include intercept)
            pass
        else:  # two sets of covariates (need to include intercept)
            features = np.hstack((features, np.ones((features.shape[0], 1))))

        # Fitting the model
        self.rasch_model = LogisticRegression()
        self.rasch_model.fit(features, labels)

        # Predicted probs
        self.gammas = self.rasch_model.mu[: self.x_dim]
        self.thetas = self.X @ self.gammas
        self.psi = self.rasch_model.mu[self.x_dim :]

        if type(X) != np.ndarray and type(Z) != np.ndarray:  # basic Rasch model (no intercept)
            self.betas = np.hstack((self.psi, np.array([0])))
            self.logits = self.thetas[:, None] + self.betas[None, :]
        elif type(X) != np.ndarray or type(Z) != np.ndarray:  # just one set of covariates (no intercept)
            self.betas = self.Z @ self.psi
            self.logits = self.thetas[:, None] + self.betas[None, :]
        else:  # two sets of covariates (intercept included)
            self.betas = self.Z @ self.psi[:-1]
            self.logits = self.thetas[:, None] + self.betas[None, :] + self.psi[-1]

    def get_Y_hat(self):
        """
        Computes the predicted probabilities.

        Returns:
            array-like: Predicted probabilities.
        """
        P_hat = sigmoid(self.logits)
        Y_hat = np.zeros(self.seen_examples.shape)
        Y_hat[self.seen_examples] = self.Y[self.seen_examples]
        Y_hat[~self.seen_examples] = P_hat[~self.seen_examples]
        return Y_hat

def EfficEval(Y_list, M_list, m=5, fisher_info=False):
    assert len(M_list)==len(Y_list)

    #Preparing data
    models = np.unique([x for xs in M_list for x in xs]).tolist()
    n_models = np.max(models)+1
    S = [np.zeros((n_models, y.shape[1])) for y in Y_list]
    Y = [np.zeros((n_models, y.shape[1])) for y in Y_list]
    for y, s, y_list, m_list in zip(Y, S, Y_list, M_list):
        y[m_list] = y_list
        s[m_list] = 1
    S = np.hstack(S).astype(bool)
    Y = np.hstack(Y)
    
    #Fitting Rasch model
    rm = ExtendedRaschModel()
    rm.fit(S, Y)
    
    #Checking which models will be re-evaluated in the next step
    if fisher_info:
        betas = rm.betas[-Y_list[-1].shape[1]:]
        centers = np.percentile(betas, np.linspace(5,95,m))[:,None]
        coreset_questions = pairwise_distances(centers, betas[:,None]).argmin(1)
        F = (sigmoid(rm.logits)*(1-sigmoid(rm.logits)))[:,coreset_questions]
        reval_models = [models[k] for k in np.unique(F[models].argmax(0)).tolist()]
    else:
        thetas = rm.thetas[models].squeeze()
        centers = np.percentile(thetas, np.linspace(0,100,m))[:,None]
        coreset_models = pairwise_distances(centers, thetas[:,None]).argmin(1)
        reval_models = [models[k] for k in np.unique(coreset_models).tolist()]
        
    #Computing Y_hats
    sample_sizes = [0]+np.cumsum([y.shape[1] for y in Y_list]).tolist()
    Y_hat_list = []
    for i in range(len(sample_sizes)-1):
        Y_hat_list.append(rm.get_Y_hat()[:,list(range(sample_sizes[i],sample_sizes[i+1]))])

    #Output
    return reval_models, Y_hat_list