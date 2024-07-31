# -----------------------------------------------------------------------------
# Functions for model training
# -----------------------------------------------------------------------------
from model import get_ccd_model
from data import create_dataloader


class Trainer:
    '''
    To construct instance of trainer object for each stage
    '''
    def __init__(self, args, model, stage_i, dataset_train, dataset_val, dataset_test):
        self.args = args
        self.stage_i = stage_i
        self.ccd_model = get_ccd_model(self.args, model, self.stage_i)
        self.train_dataloader_i = create_dataloader(self.args, dataset_train[stage_i], self.stage_i) 
        self.val_dataloader = create_dataloader(self.args, dataset_val, -1)

        if self.args.transductive_evaluation:
            self.test_dataloader_i = create_dataloader(self.args, dataset_train[:self.stage_i + 1], -2)
        else:
            self.test_dataloader_i = create_dataloader(self.args, dataset_test[:self.stage_i + 1], -2)

    def run(self):
        if self.args.train:
            model = self.ccd_model.fit(self.train_dataloader_i, self.val_dataloader)
            if self.stage_i > 0:
                self.ccd_model.eval(self.test_dataloader_i) # to acquire labels for CCD eval metric
            return model

        if self.args.test and self.stage_i > 0:
            self.ccd_model.eval(self.test_dataloader_i)
            return None


def RunContinualTrainer(args, datasets_train, datasets_val, datasets_test):
    model = None
    for stage_i in range(args.n_stage+1):
        model = Trainer(
            args, 
            model, 
            stage_i, 
            datasets_train, 
            datasets_val, 
            datasets_test
        ).run()