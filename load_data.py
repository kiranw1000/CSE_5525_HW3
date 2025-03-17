import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split
        self.nl = []
        self.sql = []
        self.queries = []
        self.tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.extra_id = "<extra_id_0>"
        self.extra_token = self.tokenizer.convert_tokens_to_ids(self.extra_id)
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        lines = load_lines(os.path.join(data_folder, f"{split}.nl"))
        if split != "test" and split != "mini_test":
            queries = load_lines(os.path.join(data_folder, f"{split}.sql"))
            self.sql = queries
        for i,line in enumerate(lines):
            self.nl.append(tokenizer(line, return_tensors='pt'))
            if split != "test" and split != "mini_test":
                self.queries.append(tokenizer(self.extra_id+queries[i], return_tensors='pt'))
    
    def __len__(self):
        return len(self.nl)

    def __getitem__(self, idx):
        if self.split == "test" or self.split == "mini_test":
            return self.nl[idx]
        return self.nl[idx], self.queries[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    temp = [batch[i][0]['input_ids'].T for i in range(len(batch))]
    encoder_ids = torch.squeeze(pad_sequence(temp, padding_value=PAD_IDX),2).mT
    encoder_mask = torch.squeeze(pad_sequence([batch[i][0]['attention_mask'].T for i in range(len(batch))], padding_value=PAD_IDX), 2).mT
    decoder_inputs = torch.squeeze(pad_sequence([batch[i][1]['input_ids'][:,:-1].T for i in range(len(batch))], padding_value=PAD_IDX), 2).mT
    decoder_targets = torch.squeeze(pad_sequence([batch[i][1]['input_ids'][:,1:].T for i in range(len(batch))], padding_value=PAD_IDX) ,2).mT
    initial_decoder_inputs = [PAD_IDX for i in range(len(batch))]
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    temp = [batch[i]['input_ids'].T for i in range(len(batch))]
    encoder_ids = torch.squeeze(pad_sequence(temp, padding_value=PAD_IDX), 2).mT
    encoder_mask = torch.squeeze(pad_sequence([batch[i]['attention_mask'].T for i in range(len(batch))], padding_value=PAD_IDX), 2).mT
    initial_decoder_inputs = torch.tensor([[PAD_IDX for i in range(len(batch))]]).mT
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train" or split == "mini_train"
    # collate_fn = normal_collate_fn if split != "test" and split!="mini_test" else test_collate_fn
    collate_fn = normal_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, mini=False):
    train_loader = get_dataloader(batch_size, f"{'mini_' if mini else ''}train")
    dev_loader = get_dataloader(test_batch_size, f"{'mini_' if mini else ''}dev")
    test_loader = get_dataloader(test_batch_size, f"{'mini_' if mini else ''}test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x