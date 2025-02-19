import copy
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics.pairwise import pairwise_distances

sigmoid = nn.Sigmoid()

def loss_matrix(Y, P, eps=1e-5):
    return -(Y*(P+eps).log() + (1-Y)*(1-P+eps).log())

def irt_forward(theta, beta):
    return sigmoid(theta.squeeze()[:,None] - beta.squeeze()[None,:])

class RaschModel:

    def __init__(self):
        """
        Initializes the Rasch model.
        """
        pass

    def fit(self,
            seen_examples,
            Y,
            max_epochs=10000,
            learning_rate=.01,
            scheduler_factor=0.1,
            scheduler_patience=10,
            earlystop_patience=50,
            tolerance=1e-5,
            random_seed=0,
            weight_decay=0,
            device='cuda'):

        """
        Fits the extended Rasch model to the data.

        Parameters:
            seen_examples (array-like): Boolean array indicating seen examples.
            Y (array-like): Target matrix.
        """

        Y = torch.tensor(Y, dtype=torch.float32, device=device)
        seen_examples = torch.tensor(seen_examples, device=device)
        
        ### Defining training variables
        parameters = []
        torch.manual_seed(random_seed)
        beta = torch.nn.Parameter(torch.normal(0, 1, size=(Y.shape[1],), dtype=torch.float32, device=device))
        theta = torch.nn.Parameter(torch.normal(0, 1, size=(Y.shape[0],), dtype=torch.float32, device=device))
        parameters.append(beta)
        parameters.append(theta)
        
        ### Training
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
        
        # Early stopping parameters
        best_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        early_stop = False
        
        # Losses
        train_losses =[]
        
        # Early stopping setup
        best_loss = float('inf')
        epochs_no_improve = 0
        
        # Loop
        for epoch in range(max_epochs):
            
            optimizer.zero_grad()
            loss = loss_matrix(Y, irt_forward(theta, beta))[seen_examples].mean() 
            loss.backward()
            train_losses.append(loss.item())
        
            optimizer.step()
            scheduler.step(loss)
            loss = loss.item()
            
            if loss < best_loss - tolerance:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= earlystop_patience:
                    break

        ### Outputs
        Y = Y.detach().cpu()
        seen_examples = seen_examples.detach().cpu()
        theta = theta.detach().cpu()
        beta = beta.detach().cpu()

        self.P_hat = sigmoid(theta[:, None] - beta[None, :])
        self.Y_hat = self.P_hat.clone()
        self.Y_hat[seen_examples] = Y[seen_examples]
        
        self.P_hat = self.P_hat.numpy()
        self.Y_hat = self.Y_hat.numpy()
        self.thetas = theta.numpy()
        self.betas = beta.numpy()
        self.train_losses = train_losses

    def get_Y_hat(self):
        """
        Computes the predicted probabilities.

        Returns:
            array-like: Predicted probabilities.
        """
        
        return self.Y_hat

def EfficEval(Y_list, M_list, m=5, fisher_info=True):
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
    rm = RaschModel()
    rm.fit(S, Y)
    
    #Checking which models will be re-evaluated in the next step
    if fisher_info:
        betas = rm.betas[:] #[-Y_list[-1].shape[1]:]
        centers = np.percentile(betas, np.linspace(5,95,m))[:,None]
        coreset_questions = pairwise_distances(centers, betas[:,None]).argmin(1)
        F = (rm.P_hat*(1-rm.P_hat))[:,coreset_questions]
        reval_models = [models[k] for k in np.unique(F[models].argmax(0)).tolist()]
        while len(reval_models)<m: #get the most different models to complete m (sometimes reval_models can have less than m model)
            rest_models = np.array([m for m in models if m not in reval_models])
            d = pairwise_distances(rm.thetas[reval_models][:,None],rm.thetas[rest_models][:,None]).mean(0)
            reval_models.append(int(rest_models[d.argmax()]))
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