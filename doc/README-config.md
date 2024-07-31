# Model configurations

This README provides guidelines on how to configure **PromptCCD** hyperparameters. The `*.yaml` can be accesed at `config/%DATASET%/`.

## Important configs
* `run_ccd`: set to true when running the model for training and testing.
* `ccd_model`: indicate which CCD model to choose.
* `manual_seed`: seed for training.
* `save_path`: indicate where to store the experiment results.
* `eval_version`: evaluation metric to evaluate the model, e.g., `ccd` or `gcd`.
* `transductive_evaluation`: please refer to supplementary material sec. **S3**.

## Data
* `dataset`: dataset to choose for experiment.
* `ccd_split_ratio`: how the dataset is splitted into `n_stage`(s) 
* `random_split_ratio`: split ratio between train and val datasets. 

## Optimization
* `epoch`: number of training for each stage. 
* `optim`: optimizer algorithm.
* `base_lr`: initial learning rate. 
* `lr_scheduler`: learning rate scheduler.
* `temperature`: temperature parameter for info ncd logits loss function.
* `sup_con_weight`: weight for the supervised contrastive loss function for each stage. (default: [0.35, 0., 0., 0.]) 

## Model & Prompt module
* `use_dinov2`: if set true, then DINOv2 model is used for training. (default: DINO)
* `grad_from_block`: DINO's transformer starting block to optimize.
* `prompt_pool`: set to true for prompt learning.
* `top_k`: top k mean components for prompting .
* `fit_gmm_every_n_epoch`: GMM optimiztion schedule. 
* `num_gmm_samples`: number of gmm samples per class to be stored after training. 
* `covariance_type`: GMM covariance types. 


