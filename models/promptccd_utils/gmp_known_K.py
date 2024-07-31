# -----------------------------------------------------------------------------
# Functions for Gaussian Mixture Model based Prompt Method with known K
# -----------------------------------------------------------------------------
import os
import logging
from tqdm import tqdm

import torch
import numpy as np
from pycave import set_logging_level
from pycave.bayes import GaussianMixture as GMM

import gc

device = torch.device('cuda')
set_logging_level(logging.WARNING)


class GMMPrompt():
    def __init__(self, args, num_components=1, train=True):
        super().__init__()
        self.args = args
        self.num_components = num_components

        # Initialize GMM model with given number of components and trainer parameters
        if train:
            self.gmm = GMM(
                num_components=int(num_components), 
                trainer_params=dict(gpus=1, enable_progress_bar=False),
                covariance_type=self.args.covariance_type,
                convergence_tolerance=self.args.convergence_tolerance,
                covariance_regularization=self.args.covariance_regularization,
            )

    def init_gmm_for_eval(self, stage_i):
        self.gmm = GMM.load(os.path.join(self.args.save_path, f'gmm/gmm_{str(stage_i)}'))

    def fit(self, args, model, data_loader, stage=0):

        all_feats = []
        # Extract all features
        for batch in tqdm(data_loader, desc='Extract feats.', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            data, label, _, mask_lab_ = batch
            data = data.pin_memory()

            with torch.no_grad():
                feats = model(data.cuda())['x'][:, 0]
        
            # Normalize features
            feats = torch.nn.functional.normalize(feats, dim=-1)

            all_feats.append(feats)

        # Concatenate all features
        feats = torch.cat(all_feats, dim=0)

        # If stage > 0, concatenate previous samples to all_feats
        if stage > 0 and args.generate_gmm_samples == True:
            prev_samples = np.load(os.path.join(self.args.save_path, f'gmm/gmm_samples_{stage-1}.npy'))
            prev_samples = torch.from_numpy(prev_samples).to(device)

            # Combine previous samples with current samples
            feats = torch.cat([feats, prev_samples], dim=0)

        # Fit to Gaussian Mixture Model
        self.gmm.fit(feats)
        # del feats

    def predict(self, batch):
        res = dict()
        # Predict batch data to its topk mixture component
        results = self.gmm.predict_proba(batch)                     # Batch, num_components
        _, idx = torch.topk(results, k=self.args.top_k, dim=1)      # Batch, top_k

        ############
        # Random pick k components
        # idx = torch.randint(0, self.num_components, (batch.shape[0], self.args.top_k)).to(device)
        ############
        idx = idx.squeeze(1)                                        # Batch
        res['batched_prompt'] = self.gmm.model_.means[idx]          # Batch, topk, emb_dim
        res['prompt_idx'] = idx
        res['total_prompt_len'] = res['batched_prompt'].shape[1]

        del results # when NOT using ranodm pick
        gc.collect()
        return res

    def sample(self, stage):
        # For each components in GMM, generate n samples
        samples = self.gmm.sample(self.args.num_gmm_samples).cpu().numpy() # num_samples, emb_dim

        # Save sample to npy
        np.save(os.path.join(self.args.save_path, f'gmm/gmm_samples_{stage}.npy'), samples)

        # Save gmm parameters for stage_i
        self.gmm.save(os.path.join(self.args.save_path, f'gmm/gmm_{stage}'))

        del samples
