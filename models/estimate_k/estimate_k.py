# -----------------------------------------------------------------------------
# Functions for estimating K using GCD method
# Please refer to the original implementation for more details
# -----------------------------------------------------------------------------
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.optimize import minimize_scalar
from functools import partial

from util.util import info

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = torch.device('cuda:0')

def test_kmeans(K, merge_test_loader, args=None, verbose=False):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    if K is None:
        K = args.labelled_data + args.num_unlabeled_classes

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    model = use_pretrained_model(args)

    print('Collating features...')
    # First extract all features
    for batch_idx, (data, label, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):

        with torch.no_grad():
            feats = model(data.cuda())

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(args.labelled_data)
                                                 else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_lab


    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                 preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    if verbose:
        print('K')
        print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi,
                                                                             labelled_ari))
        print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi,
                                                                               unlabelled_ari))

    return labelled_acc


def test_kmeans_for_scipy(K, merge_test_loader, args=None, verbose=False):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    K = int(K)

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes


    model = use_pretrained_model(args)
    
    print('Collating features...')
    # First extract all features
    for batch_idx, (data, label, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):

        with torch.no_grad():
            feats = model(data.cuda())

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(args.labelled_data)
                                                 else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    print(f'Fitting K-Means for K = {K}...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_lab
    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                 preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    print(f'K = {K}')
    print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi,
                                                                         labelled_ari))
    print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi,
                                                                           unlabelled_ari))

    return -labelled_acc


def binary_search(merge_test_loader, args, stage):

    min_classes = args.labelled_data

    # Iter 0
    big_k = args.max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    labelled_acc_big = test_kmeans(big_k, merge_test_loader, args)
    labelled_acc_small = test_kmeans(small_k, merge_test_loader, args)
    labelled_acc_middle = test_kmeans(middle_k, merge_test_loader, args)

    print(f'Iter 0: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
    all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):

        if labelled_acc_big > labelled_acc_small:

            best_acc = max(labelled_acc_middle, labelled_acc_big)

            small_k = middle_k
            labelled_acc_small = labelled_acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)

        else:

            best_acc = max(labelled_acc_middle, labelled_acc_small)
            big_k = middle_k

            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            labelled_acc_big = labelled_acc_middle

        labelled_acc_middle = test_kmeans(middle_k, merge_test_loader, args)

        print(f'Iter {i}: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
        all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    print(f'Optimal K is {best_acc_at_k} at stage {stage}')


def scipy_optimise(merge_test_loader, args, stage):

    small_k = args.labelled_data
    big_k = args.max_classes

    test_k_means_partial = partial(test_kmeans_for_scipy, merge_test_loader=merge_test_loader, args=args, verbose=True)
    res = minimize_scalar(test_k_means_partial, bounds=(small_k, big_k), method='bounded', options={'disp': True})
    print(f'Optimal K is {res.x} at stage {stage}')


def use_pretrained_model(args):
    if args.selected_pretrained_model_for_eval == 'dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        info("Use ViT base16 DINO pretrained model")
        state_dict = torch.load(args.dino_pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

    elif args.selected_pretrained_model_for_eval == 'gcd':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        info("Use ViT base16 GCD pretrained model")
        state_dict = torch.load(args.warmup_model_dir, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    model.cuda()

    return model


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size