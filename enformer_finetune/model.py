import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from enformer_pytorch import from_pretrained, Enformer
from pytorch_lightning import LightningModule
from enformer_finetune.ttt import TTTForCausalLM, TTTConfig
import pandas as pd
from scipy.stats import pearsonr


class HyperGeneModelClass(LightningModule):
    def __init__(self, lr=0.0001):
        super(HyperGeneModelClass, self).__init__()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        self.outputs.append(output.flatten(0).detach().cpu().numpy())
        self.targets.append(target.flatten(0).detach().cpu().numpy())

    def on_test_start(self) -> None:
        self.outputs = []
        self.targets = []

    def on_test_end(self) -> None:
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        pr, p = self.calculate_pearsonr(self.outputs, self.targets)

        if self.logger is not None:
            self.logger.experiment.log(dict(pearsonR=pr))
        print(f'Test PearsonR: {pr:.2f}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def calculate_pearsonr(self, x, y):
        # 计算Pearson相关系数及其p-value
        correlation, p_value = pearsonr(x, y)
        return correlation, p_value


class TTTModel(HyperGeneModelClass):
    def __init__(self, config):
        super(TTTModel, self).__init__()
        configuration = TTTConfig(**config)
        self.model = TTTForCausalLM(configuration)
        # self.projector=nn.Sequential(
        #     nn.Linear(5,1),
        #     nn.ReLU(),
        #     nn.Flatten(-2,-1),
        #     nn.LayerNorm([20000])
        # )
        # self.regressor=nn.Sequential(
        #     nn.Linear(20000,1)
        # )

    def forward(self, x, y):
        x = self.model(x - 7, labels=y)
        # x=self.projector(x.logits)
        # x=self.regressor(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, y)

        self.log('train_loss', output.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': output.loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, y)

        self.log('val_loss', output.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_loss': output.loss}


class EnformerGeneModel(HyperGeneModelClass):
    def __init__(self, pretrained=True, linear_prob=True):
        super(EnformerGeneModel, self).__init__()
        if pretrained:
            self.enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=False,
                                            local_files_only=True)
        else:

            self.enformer = Enformer.from_hparams(
                dim=1536,
                depth=11,
                heads=8,
                output_heads=dict(human=5313, mouse=1643),
                target_length=157,
            )

        for p in self.enformer.parameters():
            p.requires_grad = not linear_prob

        for p in self.enformer.trunk[-1].parameters():
            p.requires_grad = True

        ### Xpresso
        self.projector = nn.Sequential(
            nn.LayerNorm([157, 3072]),
            nn.Linear(3072, 1),
            nn.ReLU(),
            nn.Flatten(-2, -1),
            nn.BatchNorm1d(157)
        )
        self.regressor = nn.Sequential(
            nn.Linear(157, 1)
        )

        # for m in self.projector.modules():
        #     if isinstance(m,nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #
        # for m in self.regressor.modules():
        #     if isinstance(m,nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        _, x = self.enformer(x, return_embeddings=True)
        x = x.transpose(1, 2)
        x = self.projector(x)

        x = self.regressor(x)
        x = x.transpose(1, 2)

        return x


class EnformerEPIModel(HyperGeneModelClass):
    def __init__(self, pretrained=True, linear_prob=True):
        super(EnformerEPIModel, self).__init__()

        self.df = None
        if pretrained:
            self.enformer = from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=False,
                                            local_files_only=True)
        else:

            self.enformer = Enformer.from_hparams(
                dim=1536,
                depth=11,
                heads=8,
                output_heads=dict(human=5313, mouse=1643),
                target_length=157,
            )

        for p in self.enformer.parameters():
            p.requires_grad = not linear_prob

        for p in self.enformer.trunk[-1].parameters():
            p.requires_grad = True

        self.projector = nn.Sequential(
            nn.ConvTranspose1d(3072, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.regressor = nn.Conv1d(128, 4, kernel_size=1)

    def forward(self, x):
        _, x = self.enformer(x, return_embeddings=True)
        x = x.transpose(1, 2)
        x = self.projector(x)

        x = self.regressor(x)
        x = x.transpose(1, 2)

        return x

    def on_test_start(self) -> None:
        self.min_DNase_pr = 0
        self.min_CAGE_pr = 0
        self.min_H3K27ac_pr = 0
        self.min_H3K4me3_pr = 0

        self.df = pd.DataFrame(
            {
                'DNase': [],
                'CAGE': [],
                'H3K27ac': [],
                'H3K4me3': [],
            }
        )

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        pr_DNase, _ = self.calculate_pearsonr(output[:, :, 0].flatten(0).detach().cpu().numpy(),
                                              target[:, :, 0].flatten(0).detach().cpu().numpy())
        pr_CAGE, _ = self.calculate_pearsonr(output[:, :, 1].flatten(0).detach().cpu().numpy(),
                                             target[:, :, 1].flatten(0).detach().cpu().numpy())
        pr_H3K27ac, _ = self.calculate_pearsonr(output[:, :, 2].flatten(0).detach().cpu().numpy(),
                                                target[:, :, 2].flatten(0).detach().cpu().numpy())
        pr_H3K4me3, _ = self.calculate_pearsonr(output[:, :, 3].flatten(0).detach().cpu().numpy(),
                                                target[:, :, 3].flatten(0).detach().cpu().numpy())

        temp = pd.DataFrame({
            'DNase': [pr_DNase],
            'CAGE': [pr_CAGE],
            'H3K27ac': [pr_H3K27ac],
            'H3K4me3': [pr_H3K4me3],
        })
        self.df = pd.concat([self.df, temp])

        if np.isnan(pr_DNase):
            pr_DNase = 0
        if np.isnan(pr_CAGE):
            pr_CAGE = 0
        if np.isnan(pr_H3K27ac):
            pr_H3K27ac = 0
        if np.isnan(pr_H3K4me3):
            pr_H3K4me3 = 0

        self.min_DNase_pr += pr_DNase
        self.min_CAGE_pr += pr_CAGE
        self.min_H3K27ac_pr += pr_H3K27ac
        self.min_H3K4me3_pr += pr_H3K4me3

    def on_test_end(self) -> None:
        self.df.to_csv('./enformer_fold1_samples_relevance.csv')
        self.min_DNase_pr /= self.test_length
        self.min_CAGE_pr /= self.test_length
        self.min_H3K27ac_pr /= self.test_length
        self.min_H3K4me3_pr /= self.test_length

        print(f'Test PearsonR DNase: {self.min_DNase_pr:.2f}')
        print(f'Test PearsonR CAGE: {self.min_CAGE_pr:.2f}')
        print(f'Test PearsonR H3K27ac: {self.min_H3K27ac_pr:.2f}')
        print(f'Test PearsonR H3K4me3: {self.min_H3K4me3_pr:.2f}')

        if self.logger is not None:
            self.logger.experiment.log(dict(pearsonR_DNase=self.min_DNase_pr, pearsonR_CAGE=self.min_CAGE_pr,
                                            pearsonR_H3K27ac=self.min_H3K27ac_pr,
                                            pearsonR_H3K4me3e=self.min_H3K4me3_pr))
