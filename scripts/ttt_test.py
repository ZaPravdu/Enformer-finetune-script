from enformer_finetune.ttt import TTTConfig
from enformer_finetune import utils
from torch.utils.data import DataLoader, Subset
from enformer_finetune import TTTModel

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

"""
This is a script for testing TTT model on gene expression dataset.
"""

# experiment_name='GM12878-fold1-TTT'
experiment_name = 'test'
cell = 'GM12878'
cell = 'Actual_' + cell

epochs = 1
lr = 0.0001
batch_size = 2
# Initializing a TTT ttt-1b style configuration
# configuration = TTTConfig(**TTT_STANDARD_CONFIGS['1b']) is equivalent to the following
config = {
    "hidden_size": 96,
    "intermediate_size": 256,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
}

# Initializing a model from the ttt-1b style configuration
configuration = TTTConfig(**config)
model = TTTModel(config)

# See the `Caduceus` collection page on the hub for list of available models.
model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

trainset = utils.GeneDataset('./data/Xpresso/Xpresso_fold1_ensembl_train.csv', cell=cell, model_name=model_name)
validset = utils.GeneDataset('./data/Xpresso/Xpresso_fold1_ensembl_valid.csv', cell=cell, model_name=model_name)
testset = utils.GeneDataset('./data/Xpresso/Xpresso_fold1_ensembl_test.csv', cell=cell, model_name=model_name)

if experiment_name == 'test':
    trainset = Subset(trainset, range(20))
    validset = Subset(validset, range(20))
    testset = Subset(testset, range(20))
    wandb_logger = None
else:
    wandb_logger = WandbLogger(project=experiment_name, save_dir='../weight/')
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# wandb_logger=WandbLogger(project=experiment_name, save_dir='./weight/')
trainer = Trainer(callbacks=[early_stopping], max_epochs=1, accelerator='gpu', logger=wandb_logger,
                  default_root_dir=f'./weight/{experiment_name}', log_every_n_steps=1)

for e in range(epochs):
    # 创建DataLoader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    validloader = DataLoader(validset, batch_size=batch_size, drop_last=True)

    trainer.fit(model, trainloader, validloader)

testloader = DataLoader(testset, batch_size=batch_size, drop_last=True)
trainer.test(model, testloader)

print()
