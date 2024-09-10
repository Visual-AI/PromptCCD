# -----------------------------------------------------------------------------
# Functions for k estimation based on SSGMM
# Ref: Learning Semi-supervised Gaussian Mixture Models for Generalized Category Discovery: https://arxiv.org/abs/2305.06144.
# -----------------------------------------------------------------------------
from math import lgamma

import torch
import numpy as np
from torch import mvlgamma
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors

from models.promptccd_utils.kmeans import K_Means
from models.sskmeans_utils.faster_mix_k_means_pytorch import pairwise_distance


def split_and_merge_op(u_feat, l_feat, l_targets, args, index=0, stage=0, num_cluster=[0]):
    class_num = num_cluster[0]
    results = {
        'centroids': [],
        'density': [],
        'im2cluster': [],
        'pi': [],
        'cov': [],
    }

    km = K_Means(k=class_num, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=128, use_gpu=True)
    nn_sklearn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    
    if stage == 0:
        cat_feat = u_feat
        km.fit(cat_feat)
    else:
        cat_feat = torch.cat([u_feat, l_feat], dim=0)
        km.fit_mix(u_feats=u_feat, l_feats=l_feat, l_targets=l_targets)
    
    centroids = km.cluster_centers_
    labels = km.labels_.cpu()
    pred = labels

    if stage == 0:
        cat_feat, u_feat = cat_feat.cpu(), u_feat.cpu()
    else:
        cat_feat, u_feat, l_feat, l_targets = cat_feat.cpu(), u_feat.cpu(), l_feat.cpu(), l_targets.cpu()

    prior = Priors(args, class_num, cat_feat.shape[1], )
    prior.init_priors(cat_feat)

    _, counts = torch.unique(labels, return_counts=True)
    counts = counts.cpu()
    pi = counts / float(len(cat_feat))
    data_covs = compute_data_covs_hard_assignment(labels, cat_feat, class_num, centroids.cpu(), prior)

    # NOTE: the following is to update the mu and cov using a prior. Can be disabled.
    mus = prior.compute_post_mus(counts, centroids.cpu())
    covs = []
    for k in range(len(centroids)):
        feat_k = cat_feat[labels == k]
        cov_k = prior.compute_post_cov(counts[k], feat_k.mean(axis=0), data_covs[k])  # len(codes_k) == counts[k]? yes
        covs.append(cov_k)
    covs = torch.stack(covs)
    
    # NOTE: now we have mus, covs, pi, labels for the global GMM
    if stage > 0:

        sub_clusters = get_sub_cluster_with_sskmeans(u_feat, l_feat, l_targets, labels, prior, args)
        labelled_clusters = labels[:len(l_targets)].unique().cpu()

        # NOTE: now we have sub_mus, sub_assignments, we can compute split rules now
        split_decisions = []
        for class_label, items in sub_clusters:
            if class_label in labelled_clusters:
                continue
            class_indices = labels == class_label
            mu_subs, cov_subs, pi_subs, sub_assign = items
            split_decision = split_rule(cat_feat[class_indices], sub_assign, prior, mus[class_label], mu_subs, args.split_prob)
            split_decisions.append([class_label, split_decision])
        
        remain_for_merge = np.array([class_l for class_l, split_d in split_decisions if not split_d])
        remain_mus = centroids[remain_for_merge].cpu()
        remain_covs = covs[remain_for_merge]
        remain_pi = pi[remain_for_merge]

        if remain_mus.shape[0] > 1:
            merge_decisions = []
            # each cluster will only be tested for merging with the top-1 nearest cluster 
            mu_nn = nn_sklearn.fit(remain_mus)
            for remain_idx, class_label in enumerate(remain_for_merge):
                nn = mu_nn.kneighbors(centroids[class_label].reshape(1, -1).cpu(), return_distance=False)[0][1:]
                nn = nn.item()
                merge_decision = merge_rule(remain_mus[remain_idx], remain_covs[remain_idx], remain_pi[remain_idx], 
                                            remain_mus[nn], remain_covs[nn], remain_pi[nn], 
                                            cat_feat[labels == class_label], cat_feat[labels == remain_for_merge[nn]], 
                                            prior, args.merge_prob)
                merge_decisions.append([class_label, merge_decision, nn])

        # NOTE: now we have split_decisions and merge_decisions, we can update the results
        new_centroids = None
        new_centroids = centroids.cpu()

        # perform split
        for idx, (class_label, split_d) in enumerate(split_decisions):
            if split_d:
                mu_subs = sub_clusters[idx][-1][0]
                new_centroids = torch.cat((new_centroids, mu_subs))
        
        if remain_mus.shape[0] > 1:
            # perform merge
            for class_label, merge_d, nn in merge_decisions:
                if merge_d:
                    nn_class_label = remain_for_merge[nn]
                    mean_mu = (centroids[class_label] + centroids[nn_class_label]) / 2
                    new_centroids = torch.cat((new_centroids, mean_mu.reshape(1, -1).cpu()))
                
        centroids = new_centroids 

    dist = pairwise_distance(cat_feat.cpu(), centroids.cpu())
    _, pred = torch.min(dist, dim=1)

    # update densities
    dist2center = torch.sum((cat_feat.cpu() - centroids[pred].cpu()) ** 2, dim=1).sqrt()
    
    density = torch.zeros(len(centroids))
    for center_id in pred.unique():
        if (pred == center_id).nonzero().shape[0] > 1:
            item = dist2center[pred == center_id]
            density[center_id.item()] = item.mean() / np.log(len(item) + 10)

    dmax = density.max()
    for center_id in pred.unique():
        if (pred == center_id).nonzero().shape[0] <= 1:
            density[center_id.item()] = dmax
            
    density = density.cpu().numpy()

    density = density.clip(np.percentile(density, 10), np.percentile(density, 90)) #clamp extreme values for stability
    density = args.temperature * density / density.mean()  #scale the mean to temperature 
    density = torch.from_numpy(density).cuda()

    centroids = F.normalize(centroids, p=2, dim=1).cuda()
    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(pred)    
    results['pi'].append(pi)    
    results['cov'].append(covs)    
    
    return results, None


def compute_covs(codes, logits, K, mus, use_priors=True, prior=None):
    data_covs = compute_data_covs_hard_assignment(logits, codes, K, mus, prior)
    if use_priors:
        covs = []
        r = logits.sum(axis=0)
        for k in range(K):
            cov_k = prior.compute_post_cov(r[k], mus[k], data_covs[k])
            covs.append(cov_k)
        covs = torch.stack(covs)
    else:
        covs = torch.stack([torch.eye(mus.shape[1]) * data_covs[k] for k in range(K)])
    return covs


def compute_data_covs_hard_assignment(labels, codes, K, mus, prior):
    # assume to be NIW prior
    covs = []
    for k in range(K):
        codes_k = codes[labels == k]
        N_k = float(len(codes_k))
        if N_k > 0:
            cov_k = torch.matmul(
                (codes_k - mus[k].cpu().repeat(len(codes_k), 1)).T,
                (codes_k - mus[k].cpu().repeat(len(codes_k), 1)),
            )
            cov_k = cov_k / N_k
        else:
            # NOTE: deal with empty cluster
            cov_k = torch.eye(codes.shape[1]) * 0.0005
        covs.append(cov_k)
    covs = torch.stack(covs)
    return covs


def compute_data_diag_covs_hard_assignment(labels, codes, K, mus, prior):
    pass


def log_Hastings_ratio_split(alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, split_prob):
    """This function computes the log Hastings ratio for a split.

    Args:
        alpha ([float]): The alpha hyperparameter
        N_k_1 ([int]): Number of points assigned to the first subcluster
        N_k_2 ([int]): Number of points assigned to the second subcluster
        log_ll_k_1 ([float]): The log likelihood of the points in the first subcluster
        log_ll_k_2 ([float]): The log likelihood of the points in the second subcluster
        log_ll_k ([float]): The log likelihood of the points in the second subcluster
        split_prob ([type]): Probability to split a cluster even if the Hastings' ratio is not > 1

        Returns a boolean indicating whether to perform a split
    """
    N_k = N_k_1 + N_k_2
    if N_k_2 > 0 and N_k_1 > 0:
        # each subcluster is not empty
        H = (
            np.log(alpha) + lgamma(N_k_1) + log_ll_k_1 + lgamma(N_k_2) + log_ll_k_2
        ) - (lgamma(N_k) + log_ll_k)
        split_prob = split_prob or torch.exp(H)
    else:
        H = torch.zeros(1)
        split_prob = 0

    # if Hastings ratio > 1 (or 0 in log space) perform split, if not, toss a coin
    return bool(H > 0 or split_prob > torch.rand(1))


def log_Hastings_ratio_merge(alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob):
    # use log for overflows
    if N_k_1 == 0:
        lgamma_1 = 0
    else:
        lgamma_1 = lgamma(N_k_1)
    if N_k_2 == 0:
        lgamma_2 = 0
    else:
        lgamma_2 = lgamma(N_k_2)
    # Hastings ratio in log space
    N_k = N_k_1 + N_k_2
    if N_k > 0:
        H = (
            (lgamma(N_k) - (np.log(alpha) + lgamma_1 + lgamma_2))
            + (log_ll_k - (log_ll_k_1 + log_ll_k_2))
        )
    else:
        H = torch.ones(1)

    merge_prob = merge_prob or torch.exp(H)
    return bool(H > 0 or merge_prob > torch.rand(1))


def get_sub_cluster_with_sskmeans(u_feat, l_feat, l_targets, labels, prior, args,):
    # NOTE: reads cluster assignments from sskmeans, perform clustering within each cluster
    sub_clusters = []
    # only the unlabelled data will be splited or merged
    all_labels = labels[len(l_targets):].unique().cpu()
    
    for class_label in all_labels.cpu().numpy().tolist():
        mu_sub, cov_sub, pi_sub, class_sub_assign = get_sub_assign_with_one_cluster(u_feat, labels[len(l_targets):], class_label, prior)
        sub_clusters.append([class_label, (mu_sub, cov_sub, pi_sub, class_sub_assign)])
    return sub_clusters


def split_rule(feats, sub_assignment, prior, mu, mu_subs, split_prob=0.1):
    # NOTE: deal with empty clusters first, pi_sub is [0, 1], no split
    """
    feats: NxD, subset of features
    sub_assignment: N, 0 and 1 assignments
    mu: 1xD, cluster center
    mu_subs: 2xD, sub cluster centers
    return [k, bool], split the k-th cluster or not
    """
    if len(feats) <= 5:
        # small clusters will not be splited
        return False
    
    if len(feats[sub_assignment == 0]) <= 5 or len(feats[sub_assignment == 1]) <= 5:
        return False
    
    log_ll_k = prior.log_marginal_likelihood(feats, mu)
    log_ll_k1 = prior.log_marginal_likelihood(feats[sub_assignment == 0], mu_subs[0])
    log_ll_k2 = prior.log_marginal_likelihood(feats[sub_assignment == 1], mu_subs[1])
    N_k_1 = len(feats[sub_assignment == 0])
    N_k_2 = len(feats[sub_assignment == 1])
    
    return log_Hastings_ratio_split(1.0, N_k_1, N_k_2, log_ll_k1, log_ll_k2, log_ll_k, split_prob=split_prob) # 0.03
    

def merge_rule(mu1, cov1, pi1, mu2, cov2, pi2, feat1, feat2, prior=None, merge_prob=0.1):
    all_feat = torch.cat([feat1, feat2], dim=0)
    N_k_1 = feat1.shape[0]
    N_k_2 = feat2.shape[0]
    N_k = feat1.shape[0] + feat2.shape[0]
    
    if N_k > 0:
        mus_mean = (N_k_1 / N_k) * mu1 + (N_k_2 / N_k) * mu2
    else:
        # in case both are empty clusters
        mus_mean = torch.mean(torch.stack([mu1, mu2]), axis=0)
    if prior is None:
        raise NotImplementedError
    else:
        log_ll_k = prior.log_marginal_likelihood(all_feat, mus_mean)
        log_ll_k_1 = prior.log_marginal_likelihood(feat1, mu1)
        log_ll_k_2 = prior.log_marginal_likelihood(feat2, mu2)
        
    return log_Hastings_ratio_merge(1.0, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob=merge_prob) # 0.03


def get_sub_assign_with_one_cluster(feat, labels, k, prior):
    counts = []
    class_indices = labels == k
    class_sub_feat = feat[class_indices]
    
    if len(class_sub_feat) <= 2:
        c = torch.tensor([0, len(class_sub_feat)])
        class_sub_assign = torch.ones(len(class_sub_feat), dtype=torch.long)
        mu_subs = torch.mean(class_sub_feat, dim=0, keepdim=True)
        mu_subs = torch.cat([torch.zeros_like(mu_subs), mu_subs], dim=0)
        # NOTE: empty sub clusters
    else:
        km = K_Means(k=2, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=128, use_gpu=True)
        km.fit(class_sub_feat)
        class_sub_assign = km.labels_.cpu()
        mu_subs = km.cluster_centers_
        _, c = torch.unique(class_sub_assign, return_counts=True)
    counts.extend(c.cpu().numpy().tolist())

    data_covs_sub = compute_data_covs_hard_assignment(class_sub_assign, class_sub_feat, 2, mu_subs.cpu(), prior)
    
    # update prior
    mu_subs = prior.compute_post_mus(torch.tensor(counts), mu_subs.cpu())
    covs_sub = []
    for k in range(2):
        covs_sub_k = prior.compute_post_cov(counts[k], class_sub_feat[class_sub_assign == k].mean(axis=0), data_covs_sub[k])
        covs_sub.append(covs_sub_k)
    covs_sub = torch.stack(covs_sub)
    
    pi_sub = torch.tensor(counts) / float(len(class_sub_feat))
    return mu_subs, covs_sub, pi_sub, class_sub_assign


class Priors:
    '''
    A prior that will hold the priors for all the parameters.
    Created on March 2022
    Copyright (c) 2022 Meitar Ronen
    '''
    def __init__(self, args, K, codes_dim, counts=10, prior_sigma_scale=None):
        self.name = "prior_class"
        self.pi_prior_type = args.pi_prior # uniform
        if args.pi_prior:
            self.pi_prior = Dirichlet_prior(K, args.pi_prior, counts)
        else:
            self.pi_prior = None
        self.mus_covs_prior = NIW_prior(args, prior_sigma_scale)

        self.name = self.mus_covs_prior.name
        self.pi_counts = args.prior_dir_counts # 0.1

    def update_pi_prior(self, K_new, counts=10, pi_prior=None):
        # pi_prior = None- keep the same pi_prior type
        if self.pi_prior:
            if pi_prior:
                self.pi_prioir = Dirichlet_prior(K_new, pi_prior, counts)
            self.pi_prior = Dirichlet_prior(K_new, self.pi_prior_type, counts)

    def comp_post_counts(self, counts):
        if self.pi_prior:
            return self.pi_prior.comp_post_counts(counts)
        else:
            return counts

    def comp_post_pi(self, pi):
        if self.pi_prior:
            return self.pi_prior.comp_post_pi(pi, self.pi_counts)
        else:
            return pi

    def get_sum_counts(self):
        return self.pi_prior.get_sum_counts()

    def init_priors(self, codes):
        return self.mus_covs_prior.init_priors(codes)

    def compute_params_post(self, codes_k, mu_k):
        return self.mus_covs_prior.compute_params_post(codes_k, mu_k)

    def compute_post_mus(self, N_ks, data_mus):
        return self.mus_covs_prior.compute_post_mus(N_ks, data_mus)

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        return self.mus_covs_prior.compute_post_cov(N_k, mu_k, data_cov_k)

    def log_marginal_likelihood(self, codes_k, mu_k):
        return self.mus_covs_prior.log_marginal_likelihood(codes_k, mu_k)


class Dirichlet_prior:
    def __init__(self, K, pi_prior="uniform", counts=10):
        self.name = "Dirichlet_dist"
        self.K = K
        self.counts = counts
        if pi_prior == "uniform":
            self.p_counts = torch.ones(K) * counts
            self.pi = self.p_counts / float(K * counts)

    def comp_post_counts(self, counts=None):
        if counts is None:
            counts = self.counts
        return counts + self.p_counts

    def comp_post_pi(self, pi, counts=None):
        if counts is None:
            # counts = 0.001
            counts = 0.1
        return (pi + counts) / (pi + counts).sum()

    def get_sum_counts(self):
        return self.K * self.counts


class NIW_prior:
    """A class used to store niw parameters and compute posteriors.
    Used as a class in case we will want to update these parameters.
    """

    def __init__(self, args, prior_sigma_scale=None):
        self.name = "NIW"
        self.prior_mu_0_choice = args.prior_mu_0 # data_mean
        self.prior_sigma_choice = args.prior_sigma_choice # isotropic
        self.prior_sigma_scale = prior_sigma_scale or args.prior_sigma_scale #  .005
        self.niw_kappa = args.prior_kappa # 0.0001
        self.niw_nu = args.prior_nu # at least feat_dim + 1
        

    def init_priors(self, codes):
        if self.prior_mu_0_choice == "data_mean":
            self.niw_m = codes.mean(axis=0)
        if self.prior_sigma_choice == "isotropic":
            self.niw_psi = (torch.eye(codes.shape[1]) * self.prior_sigma_scale).double()
        elif self.prior_sigma_choice == "data_std":
            self.niw_psi = (torch.diag(codes.std(axis=0)) * self.prior_sigma_scale).double()
        else:
            raise NotImplementedError()
        return self.niw_m, self.niw_psi

    def compute_params_post(self, codes_k, mu_k):
        # This is in HARD assignment.
        N_k = len(codes_k)
        sum_k = codes_k.sum(axis=0)
        kappa_star = self.niw_kappa + N_k
        nu_star = self.niw_nu + N_k
        mu_0_star = (self.niw_m * self.niw_kappa + sum_k) / kappa_star
        codes_minus_mu = codes_k - mu_k
        S = codes_minus_mu.T @ codes_minus_mu
        psi_star = (
            self.niw_psi
            + S
            + (self.niw_kappa * N_k / kappa_star)
            * (mu_k - self.niw_m).unsqueeze(1)
            @ (mu_k - self.niw_m).unsqueeze(0)
        )
        return kappa_star, nu_star, mu_0_star, psi_star

    def compute_post_mus(self, N_ks, data_mus):
        # N_k is the number of points in cluster K for hard assignment, and the sum of all responses to the K-th cluster for soft assignment
        return ((N_ks.reshape(-1, 1) * data_mus) + (self.niw_kappa * self.niw_m)) / (
            N_ks.reshape(-1, 1) + self.niw_kappa
        )

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)
        D = len(mu_k)
        if N_k > 0:
            return (
                self.niw_psi
                + data_cov_k * N_k  # unnormalize
                + (
                    ((self.niw_kappa * N_k) / (self.niw_kappa + N_k))
                    * ((mu_k - self.niw_m).unsqueeze(1) * (mu_k - self.niw_m).unsqueeze(0))
                )
            ) / (self.niw_nu + N_k + D + 2)
        else:
            return self.niw_psi

    def log_marginal_likelihood(self, codes_k, mu_k):
        kappa_star, nu_star, mu_0_star, psi_star = self.compute_params_post(
            codes_k, mu_k
        )
        codes_k = codes_k.double()
        (N_k, D) = codes_k.size()
        return (
            -(N_k * D / 2.0) * np.log(np.pi)
            + mvlgamma(torch.tensor(nu_star / 2.0), D)
            - mvlgamma(torch.tensor(self.niw_nu) / 2.0, D)
            + (self.niw_nu / 2.0) * torch.logdet(self.niw_psi)
            - (nu_star / 2.0) * torch.logdet(psi_star)
            + (D / 2.0) * (np.log(self.niw_kappa) - np.log(kappa_star))
        )