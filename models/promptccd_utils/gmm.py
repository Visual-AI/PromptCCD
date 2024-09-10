from math import pi

import torch
import numpy as np


def dump(gmm_model, path):
    gmm_model.save(path)


def load(path, mode="full"):
    state = torch.load(path)
    model = GaussianMixture(state["n_components"], state["n_features"], mode, eps=state["eps"])
    model.load(path)
    return model


def full_to_diagonal_covariance(full_covariance):
    # full_covariance is expected to be of shape (1, k, d, d)
    _, k, d, _ = full_covariance.shape
    
    # Initialize the diagonal covariance matrix with shape (1, k, d)
    diagonal_covariance = torch.zeros((1, k, d))
    
    # Extract the diagonal elements for each component
    for i in range(k):
        diagonal_covariance[0, i, :] = torch.diagonal(full_covariance[0, i, :, :])
    
    return diagonal_covariance


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, covariance_type="full", init_params='kmeans', mu_init=None, var_init=None, eps=1.e-6):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:              torch.Tensor (n, 1, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d) or (1, k, d, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: float
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d) or (1, k, d, d)
            eps:            float
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params
        assert self.covariance_type in ['diag']
        assert self.init_params in ["kmeans", 'random']

        self._init_params()

        self.use_float32 = False
        self.ema_step = 1

    def set_use_float32(self, state):
        self.use_float32 = state

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                # (1, k, d)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                # (1, k, d, d)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False,)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features,dtype=torch.float64).reshape(1, 1, self.n_features, self.n_features).repeat(1,self.n_components,1, 1),
                    requires_grad=False)

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1./self.n_components)

        self.params_fitted = False
        # train mode
        self.train = True

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, mean_data=False).mean() * n + free_params * np.log(n)

        return bic

    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans":
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes inane values, reinitialize model
                self.__init__(self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data,_ = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True
        return self

    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, mean_data=False)
        return score
    
    def fair_score_samples(self, x):
        x = self.check_size(x)

        score = self.__fair_score(x, mean_data=False)
        return score

    def _cal_mutmal_x_cov(self, mat_a, mat_b):
        """
        cal x mutmal covriance without use mutmal to reduce memory
        mat_a:torch.Tensor (n,k,1,d)
        mat_b:torch.Tensor (1,k,d,d)
        """
        res = torch.zeros(mat_a.shape,dtype=mat_a.dtype, device=mat_a.device)
        for i in range(self.n_components):
            mat_a_i = mat_a[:,i,:,:].squeeze(-2)
            mat_b_i = mat_b[0,i,:,:].squeeze()
            res[:,i,:,:] = mat_a_i.mm(mat_b_i).unsqueeze(1)
        return res

    def _cal_mutmal_x_x(self, mat_a, mat_b):
        """
        cal x mutmal x without use mutmal to reduce memory
        mat_a:torch.Tensor (n,k,1,d)
        mat_b:torch.Tensor (n,k,d,1)
        """
        return torch.sum(mat_a.squeeze(-2)*mat_b.squeeze(-1),dim=2,keepdim=True)

    def _cal_log_det(self, var):
        """
        cal log_det in log space, which can prevent overflow
        var: torch.Tensor (1,k,d,)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)
        for k in range(self.n_components):
            evals, evecs = torch.linalg.eig(var[0,k])
            log_det[k] = torch.log(evals.real).sum()
        return log_det.unsqueeze(-1)

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "diag":
            var = full_to_diagonal_covariance(self.var)
            mu = self.mu
            prec = torch.rsqrt(var).cuda()

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)

            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det
    
        else:
            raise ValueError("covariance type not supported")

    def _fair_estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            assert False

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p)

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n,d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """cov_sum
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        def _cal_var_serial(x, mu, resp, batch_size=1000):
            n, k, d = x.shape[0], resp.shape[1] ,x.shape[-1]
            cov_sum = torch.zeros(size=(1,k,d,d), dtype=torch.float64, device=x.device)#.detach()
            steps = n // batch_size + (0 if n % batch_size == 0 else 1)
            for i in range(steps):
                x_mu = (x - mu)[i*(batch_size):(i+1)*batch_size,...].unsqueeze(-1)
                x_mu_T = (x - mu)[i*(batch_size):(i+1)*batch_size,...].unsqueeze(-2)
                cov_sum += torch.sum(x_mu.matmul(x_mu_T) * resp[i*(batch_size):(i+1)*batch_size,...].unsqueeze(-1), dim=0,
                            keepdim=True)#.detach()
            return cov_sum / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps

        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            # var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
            #                keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps
            var = _cal_var_serial(x, mu, resp, batch_size=128)
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]
        #torch.cuda.empty_cache()

        return pi, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, mean_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            mean_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if mean_data:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)
    
    def __fair_score(self, x, mean_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            mean_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._fair_estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if mean_data:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def __max_score(self, x):
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        return weighted_log_prob.max(dim=1)[0]


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """

        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """

        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi

    @staticmethod
    def get_kmeans_mu(X, n_centers, min_delta=1e-3, init_times=20):
        """
        input:
        x:              torch.Tensor (n, d),(n, 1, d)
        output:
        center:         torch.Tensor (1, k, d)
        """
        X = X.squeeze(1)
        min, max = X.min(), X.max()
        X = (X-min) / (max - min)
        min_cost = np.inf
        for i in range(init_times):
            tmp_center = X[np.random.choice(np.arange(X.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((X.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            cost = 0
            for c in range(n_centers):
                cost += torch.norm(X[l2_cls==c] - tmp_center[c],p=2,dim=1).sum()
            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        # center = X[np.random.choice(np.arange(X.shape[0]), size=4, replace=False), ...]
        delta = np.inf
        while delta > min_delta:
            l2_dis = torch.norm((X.unsqueeze(1).repeat(1,n_centers,1)-center),p=2,dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()
            for c in range(n_centers):
                center[c] = X[l2_cls==c].mean(dim=0)
            delta = torch.norm((center_old-center),dim=1).max()

        return (center.unsqueeze(0)*(max-min)+min)

    def score(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).cuda()
        return self.score_samples(x)
    
    def fair_score(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).cuda()
        x = self.check_size(x)
        return self.fair_score_samples(x)
        

    def negative_score(self, x):
        return -self.score(x)
    
    def negative_score_max(self, x):
        return -self.__max_score(x)

    def save(self, path):
        state = {"n_components": self.n_components,
                 "n_features": self.n_features,
                 "eps": self.eps,
                 "state_dict": self.state_dict(),
                 "params_fitted":self.params_fitted}
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.load_state_dict(state["state_dict"])
        self.n_components = state["n_components"]
        self.n_features = state["n_features"]
        self.eps = state["eps"]
        self.params_fitted = state["params_fitted"]
    
    def train(self):
        self.train=True
    
    def eval(self):
        self.train=False  
        self.inverse_var = torch.inverse(self.var)
        self.log_det = self._cal_log_det(self.var)
        if self.use_float32:
            self.inverse_var = self.inverse_var.float()
            self.log_det = self.log_det.float()  

    def set_weight(self, weight):
        self.weight = weight

    def ema_update(self, gmm, decay=0.99):

        self.ema_step += 1
        decay = decay if self.params_fitted else 0.0
        for p_self, p_gmm in zip(self.parameters(), gmm.parameters()):
            p_self.data.mul_(decay).add_(p_gmm.data*(1-decay))

        # match_map = {}
        # for current_mu_idx in range(self.n_components):
        #     for target_mu_idx in range(self.n_components):
        #         if target_mu_idx in match_map.values():
        #             continue
        #         if current_mu_idx not in match_map.keys():
        #             match_map[current_mu_idx] = target_mu_idx
        #             continue
        #         current_distance = torch.norm(self.mu[0,current_mu_idx]-gmm.mu[0,target_mu_idx],2)
        #         min_distance = torch.norm(self.mu[0,current_mu_idx]-match_map[current_mu_idx],2)
        #         if current_distance < min_distance:
        #             match_map[current_mu_idx] = target_mu_idx
        
        # # update pi
        # for i in range(self.n_components):
        #     self.pi[:,i,:].mul_(decay).add_(gmm.pi[:,match_map[i],:]*(1-decay))

        # # update mu
        # for i in range(self.n_components):
        #     self.mu[:,i,:].mul_(decay).add_(gmm.mu[:,match_map[i],:]*(1-decay))

        # # # update var
        # for i in range(self.n_components):
        #     self.var[:,i,:].mul_(decay).add_(gmm.var[:,match_map[i],:]*(1-decay))
        
        self.params_fitted = True

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        mu = self.mu.squeeze(0).cpu().numpy()
        var = self.var.squeeze(0).cpu().numpy()
        pi = self.pi.squeeze(0)
        pi = pi.squeeze(-1).cpu().numpy()


        components, n_features = mu.shape
        n_samples_comp = np.array([n_samples]*components)

        if self.covariance_type == "full":
            X = np.vstack(
                [
                    np.random.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        mu, var, n_samples_comp
                    )
                ]
            )
        elif self.covariance_type == "tied":
            X = np.vstack(
                [
                    np.random.multivariate_normal(mean, var, int(sample))
                    for (mean, sample) in zip(mu, n_samples_comp)
                ]
            )
        else:
            var = full_to_diagonal_covariance(self.var).squeeze(0).cpu().numpy()
            X = np.vstack(
                [
                    mean
                    + np.random.standard_normal(size=(sample, n_features))
                    * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                       mu, var, n_samples_comp
                    )
                ]
            )

        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)


@torch.no_grad()
def get_kmeans_mu(X, K, min_delta=1e-6):
    """
    input:
    x:              torch.Tensor (n, d)
    output:
    center:         torch.Tensor (1, k, d)
    """
    min,max = X.min(), X.max()
    X = (X-min) / (max - min)
    center = X[np.random.randint(0, X.shape[0], size=K), ...]
    delta = np.inf
    while delta > min_delta:
        l2_dis = torch.norm((X.unsqueeze(1).repeat(1,K,1)-center),p=2,dim=2)
        l2_cls = torch.argmin(l2_dis, dim=1)
        center_old = center.clone()
        for c in range(K):
            center[c] = X[l2_cls==c].mean(dim=0)
        delta = torch.norm((center_old-center),dim=1).max()
    return (center.unsqueeze(0)*(max-min)+min)



