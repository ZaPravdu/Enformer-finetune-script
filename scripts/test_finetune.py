from enformer_finetune import utils
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from enformer_finetune import EnformerEPIModel


experiment_name = 'K562-fold1-Ph-epi'
model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
testset = utils.EPIDataset('../data/K562_EPI/K562_EPI_fold1_ensembl_test.csv', model_name=model_name)
# testset=Subset(testset,range(20))


model = EnformerEPIModel.load_from_checkpoint(
    './weight/K562-fold1-epi-Enformer/3tv2spmy/checkpoints/epoch=0-step=3267.ckpt')
# model=cfx.CaduceusFinetuneModel.load_from_checkpoint('./weight/K562-fold1-Ph-epi/07pqy2xy/checkpoints/epoch=0-step=3267.ckpt',
#                                                      model_name=model_name,
#                                                      output_hidden_states=True)
# model.load_state_dict(torch.load('./weight/K562-fold1-Ph-corrected/rvuks1af/checkpoints/epoch=0-step=3275.ckpt'))
# model.load_from_checkpoint('./weight/K562-fold1-Ph-corrected/rvuks1af/checkpoints/epoch=0-step=3275.ckpt')
for p in model.parameters():
    p.requires_grad = False

trainer = Trainer(max_epochs=1, accelerator='gpu', default_root_dir=f'./weight/{experiment_name}', log_every_n_steps=1, logger=None)
testloader = DataLoader(testset, batch_size=1, drop_last=True)
model.test_length = len(testset)
trainer.test(model, testloader)
