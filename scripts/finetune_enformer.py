from enformer_finetune import utils
from torch.utils.data import DataLoader, Subset
from enformer_finetune import EnformerEPIModel

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import os

# os.chdir('../')

cell = 'K562'

# experiment_name = f'{cell}-fold1-epi-Enformer-corrected'
experiment_name='test'
epochs = 1
lr = 0.0001
batch_size = 4
cell = 'Actual_' + cell

# See the `Caduceus` collection page on the hub for list of available models.
model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
model = EnformerEPIModel()

trainset = utils.EPIDataset('./data/K562_EPI/K562_EPI_fold1_ensembl_train.csv', cell=cell, model_name=model_name,
                            data_augment=0.5)
validset = utils.EPIDataset('./data/K562_EPI/K562_EPI_fold1_ensembl_valid.csv', cell=cell, model_name=model_name)
testset = utils.EPIDataset('./data/K562_EPI/K562_EPI_fold1_ensembl_test.csv', cell=cell, model_name=model_name)

if experiment_name == 'test':
    trainset = Subset(trainset, range(20))
    validset = Subset(validset, range(20))
    testset = Subset(testset, range(20))
    wandb_logger = None
else:
    wandb_logger = WandbLogger(project=experiment_name, save_dir='./weight/')

checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控验证集损失
        dirpath=f'./weight/{experiment_name}',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

trainer = Trainer(callbacks=[early_stopping, checkpoint_callback], max_epochs=1, accelerator='gpu', logger=wandb_logger,
                  default_root_dir=f'./weight/{experiment_name}', log_every_n_steps=1)

model.cuda()

for e in range(epochs):
    # 创建DataLoader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    validloader = DataLoader(validset, batch_size=batch_size, drop_last=True)

    trainer.fit(model, trainloader, validloader)
model.test_length = len(testset)
testloader = DataLoader(testset, batch_size=1, drop_last=True)
trainer.test(model, testloader)

print()
