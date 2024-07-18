import torch
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
from pyfaidx import Fasta
import sys
import numpy as np
import ast

from pytorch_lightning.callbacks import TQDMProgressBar
import torch.nn.functional as F
import math
import h5py
import random
from Bio.Seq import Seq

MAX_ALLOWED_LENGTH = 2 ** 20


class GeneDataset(torch.utils.data.Dataset):
    """Loop through bed file, retrieve (chr, start, end), query fasta file for sequence."""

    def __init__(
            self,
            csv_path,
            cell=None,
            tokenizer=None,
            add_eos=False,
            return_augs=False,
            model_name=None,
            mlm=False,
            mlm_probability=None,
            one_hot=False
    ):
        self.one_hot = one_hot
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.data = pd.read_csv(csv_path, usecols=[cell, 'sequence_20kb_Ensembl'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        target, seq = self.data.iloc[idx]

        seq = self.tokenizer(seq, truncation=True, add_special_tokens=False)

        # convert to tensor
        seq = torch.LongTensor(seq['input_ids'])

        if not isinstance(target, torch.Tensor):
            target = torch.Tensor([target])

        # replace N token with a pad token, so we can ignore it in the loss
        # seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int["N"], self.tokenizer.pad_token_id)

        return seq, target


class EPIDataset(torch.utils.data.Dataset):
    """
    A dataset class used to read epi dataset. I put sequences in a csv file and targets in a h5 file. You can customize this class.
    """

    def __init__(
            self,
            csv_path,
            cell=None,
            tokenizer=None,
            add_eos=False,
            return_augs=False,
            model_name=None,
            mlm=False,
            mlm_probability=None,
            one_hot=False,
            data_augment=None
    ):
        self.one_hot = one_hot
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.cell = cell
        self.data = pd.read_csv(csv_path)

        self.data_augment = data_augment
        self.np_idx = self.data['Unnamed: 0.1']

        with h5py.File('./data/K562_EPI/epi.h5', 'r') as f:
            # 获取数据集
            dataset_name = 'target'
            if dataset_name in f:
                targets = f[dataset_name]

                # 将数据集内容读取为NumPy数组
                self.targets = targets[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        """Returns a sequence of specified len"""
        # sample a random row from df
        row = self.data.iloc[idx]

        target = self.targets[self.np_idx[idx]]

        seq = row['seq']

        # target=np.stack([DNase_profile, CAGE_profile, H3K27ac_profile, H3K4me3_profile])
        if not isinstance(target, torch.Tensor):
            target = torch.Tensor(target)

        if self.data_augment:
            if random.random() < self.data_augment:
                seq = str(Seq(seq).reverse_complement())
                target = target.flip([1])

        seq = self.tokenizer(seq, truncation=True, add_special_tokens=False)

        # convert to tensor
        seq = torch.LongTensor(seq['input_ids'])
        target = torch.log(target + 1)

        return seq, target.transpose(0, 1)

    def process_signal(self, start_i, profile, bw, chrom):
        end_i = start_i + 64
        max = bw.chroms()[chrom]
        if end_i > max:
            if start_i > max:
                signal = 0
            else:
                signal = bw.stats(chrom, start_i, max, type="sum")[0]
        else:
            signal = bw.stats(chrom, start_i, end_i, type="sum")[0]

        profile.append(signal)
        return profile


if __name__ == '__main__':
    testset = GeneDataset('~/caduceus/data/Xpresso/xpresso_data_for_training_caduecus.csv')
    print(testset[0])
    print(len(testset))
