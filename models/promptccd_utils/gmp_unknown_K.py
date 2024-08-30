# -----------------------------------------------------------------------------
# Functions for Gaussian Mixture Model based Prompt Method with unknown K
# -----------------------------------------------------------------------------
import os

import torch
import numpy as np

from models.promptccd_utils.gmm import GaussianMixture as GMM
from models.promptccd_utils.split_and_merge import compute_covs

device = torch.device('cuda')


class GMMPrompt():
    def __init__(self, args, num_components=1, train=True):
        super().__init__()
        self.args = args

        # Initialize GMM model with given number of components and trainer parameters
        if train:
            self.gmm = GMM(
                n_components=int(num_components), 
                n_features=args.feat_dim,
                covariance_type=args.covariance_type,
                eps=args.covariance_regularization,
            )

    def init_gmm_for_eval(self, stage_i):
        self.gmm = GMM.load(os.path.join(self.args.save_path, f'gmm/gmm_{str(stage_i)}'))

    def _estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        """Estimate the diagonal covariance vectors.

        Parameters
        ----------
        responsibilities : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features)
            The covariance vector of the current components.
        """
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means**2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
    
    def reparameterize(self, cluster_results, cat_feats):

        # Update number of predicted micture components
        self.gmm.n_components = len(cluster_results['centroids'][0])

        # Update GMM's mixture variances
        self.gmm.var.data = compute_covs(
            cat_feats.cpu(),
            cluster_results['im2cluster'][0],
            self.gmm.n_components,
            cluster_results['centroids'][0].cpu(),
            use_priors=False,
            prior=None
        ).unsqueeze(0).cuda()

        # Update GMM's mixture means
        self.gmm.mu.data = cluster_results['centroids'][0].unsqueeze(0).cuda()

        pred = cluster_results['im2cluster'][0]
        not_in_pred = []
        for i in range(len(cluster_results['centroids'][0])):
            if i not in pred.unique():
                not_in_pred.append(i)
        _, counts = torch.unique(pred, return_counts=True)
        counts = counts.tolist()
        for i in not_in_pred:
            counts.insert(i, 0)
        counts = torch.tensor(counts)

        # Update GMM's mixture weights
        pi = (counts / float(len(cluster_results['im2cluster'][0]))).unsqueeze(0)
        self.gmm.pi.data = pi.unsqueeze(-1).cuda()

    def predict(self, batch):
        res = dict()
        # Predict batch data to its topk mixture component
        results = self.gmm.predict_proba(batch)                 # Batch, num_components
        _, idx = torch.topk(results, k=self.args.top_k, dim=1)  # Batch, top_k
        idx = idx.squeeze(1)                                    # Batch
        mu = self.gmm.mu.squeeze(0)                             # num_components, emb_dim
        res['batched_prompt'] = mu[idx].float()                 # Batch, topk, emb_dim
        res['prompt_idx'] = idx
        res['total_prompt_len'] = res['batched_prompt'].shape[1]

        del results
        return res

    def sample(self, stage):
        # For each components in GMM, generate n samples
        samples, label = self.gmm.sample(self.args.num_gmm_samples) # num_samples, emb_dim

        # Save sample to npy
        np.save(os.path.join(self.args.save_path, f'gmm/gmm_samples_{stage}.npy'), samples)

        # Save labels to npy
        np.save(os.path.join(self.args.save_path, f'gmm/gmm_pseudo_labels_{stage}.npy'), label)

        # Save gmm parameters for stage_i
        self.gmm.save(os.path.join(self.args.save_path, f'gmm/gmm_{stage}'))

        del samples
