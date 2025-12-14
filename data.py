import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np

class WordDataset(Dataset):
    def __init__(self, sequences, seq_len=5):
        self.samples = []
        for sent in sequences:
            for i in range(1, len(sent)):
                x = sent[:i]
                y = sent[i]
                x = [0] * (seq_len - len(x)) + x[-seq_len:]
                self.samples.append((x, y))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

class ClassificationDataset(Dataset):
    def __init__(self, sequences, labels, seq_len=5, unk_idx=None, unk_prob=0.0):
        """
        Args:
            sequences: list of tokenized sequences
            labels: list of labels
            seq_len: maximum sequence length (pad/truncate to this length)
            unk_idx: index of the <unk> token in the vocabulary
            unk_prob: probability of replacing a token with <unk>
        """
        self.samples = []
        self.unk_idx = unk_idx
        self.unk_prob = unk_prob

        for sent, label in zip(sequences, labels):
            # Pad/truncate
            x = sent[-seq_len:]
            x = [0] * (seq_len - len(x)) + x
            self.samples.append((x, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        if self.unk_idx is not None and self.unk_prob > 0:
            x = [
                self.unk_idx if random.random() < self.unk_prob else token
                for token in x
            ]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y)

#################################################
#### Toy Sentences Language Modeling Dataset ####
#################################################

with open("./data/TS_LM_data_train.txt", "r") as f:
    LM_train_sentences = f.readlines()

with open("./data/TS_LM_data_test.txt", "r") as f:
    LM_test_sentences = f.readlines()

tokenized_train = [s.lower().split() for s in LM_train_sentences]
tokenized_test = [s.lower().split() for s in LM_test_sentences]
LM_vocab = sorted(set(w for s in tokenized_train for w in s))
LM_vocab = ["<pad>", "<unk>"] + LM_vocab
LM_word2idx = {w: i for i, w in enumerate(LM_vocab)}
LM_idx2word = {i: w for w, i in LM_word2idx.items()}

def encode_sentence(sent):
    return [LM_word2idx.get(w, LM_word2idx["<unk>"]) for w in sent]

LM_train_data = [encode_sentence(s) for s in tokenized_train]
LM_test_data = [encode_sentence(s) for s in tokenized_test]

# LM_seq_len = 6
# batch_size = 32
# LM_train_loader = DataLoader(WordDataset(LM_train_data, LM_seq_len), batch_size=batch_size, shuffle=True, drop_last=True)
# LM_test_loader = DataLoader(WordDataset(LM_test_data, LM_seq_len), batch_size=batch_size)
# len(WordDataset(LM_train_data, LM_seq_len)), len(LM_train_loader)

#################################################
####   Meaning Classification (MC) Dataset   ####
#################################################

with open('./data/mc_train_data.txt') as f:
    MC_training_data_raw = f.readlines()

with open('./data/mc_dev_data.txt') as f:
    MC_dev_data_raw = f.readlines()

with open('./data/mc_test_data.txt') as f:
    MC_testing_data_raw = f.readlines()

vocab = dict()          # dictionary to be filled with the vocabulary in the form { word : POStag }
data = dict()           # dictionary to be filled with all the data (train, dev and test subsets); entries of the 
                        # form { sentence : label } with label encoding '1' as [1.0, 0.0] and '0' as [0.0, 1.0]
MC_training_data = []      # list of sentences in the train dataset as strings "word1 word2 ..."
MC_training_labels = []     # list of labels for the train dataset
MC_dev_data = []           # list of sentences in the dev dataset as strings "word1 word2 ..."
MC_dev_labels = []         # list of labels for the dev dataset
MC_testing_data = []       # list of sentences in the test dataset as strings "word1 word2 ..."
MC_testing_labels = []     # list of labels for the test dataset

# Go through the train data
for sent in MC_training_data_raw:
    words = sent[2:].split() 
    sent_untagged = ''
    for word in words:
        word_untagged, tag = word.split('_')
        vocab[word_untagged] = tag
        sent_untagged += word_untagged + ' '
    sentence = sent_untagged[:-1]
    MC_training_data.append(sentence)
    MC_training_labels.append(int(sent[0]))
    label = np.array([1.0, 0.0]) if sent[0] == '1' else np.array([0.0, 1.0])
    data[sentence] = label

# Go through the dev data
for sent in MC_dev_data_raw:
    words = sent[2:].split() 
    sent_untagged = ''
    for word in words:
        word_untagged, tag = word.split('_')
        vocab[word_untagged] = tag
        sent_untagged += word_untagged + ' '
    sentence = sent_untagged[:-1]
    MC_dev_data.append(sentence)
    MC_dev_labels.append(int(sent[0]))
    label = np.array([1.0, 0.0]) if sent[0] == '1' else np.array([0.0, 1.0])
    data[sentence] = label
    
# Go through the test data
for sent in MC_testing_data_raw:
    words = sent[2:].split() 
    sent_untagged = ''
    for word in words:
        word_untagged, tag = word.split('_')
        vocab[word_untagged] = tag
        sent_untagged += word_untagged + ' '
    sentence = sent_untagged[:-1]
    MC_testing_data.append(sentence)
    MC_testing_labels.append(int(sent[0]))
    label = np.array([1.0, 0.0]) if sent[0] == '1' else np.array([0.0, 1.0])
    data[sentence] = label

MC_seq_len = max(len(sent.split()) for sent in MC_training_data + MC_dev_data + MC_testing_data)

MC_tokenized_train = [s.lower().split() for s in MC_training_data]
MC_tokenized_test = [s.lower().split() for s in MC_testing_data]
MC_vocab = sorted(set(w for s in MC_tokenized_train for w in s))
MC_vocab = ["<pad>", "<unk>"] + MC_vocab
MC_word2idx = {w: i for i, w in enumerate(MC_vocab)}
MC_idx2word = {i: w for w, i in MC_word2idx.items()}

def encode_sentence(sent):
    return [MC_word2idx.get(w, MC_word2idx["<unk>"]) for w in sent]

MC_train_data = [encode_sentence(s) for s in MC_tokenized_train]
MC_test_data = [encode_sentence(s) for s in MC_tokenized_test]

MC_training_labels = [float(l) for l in MC_training_labels]
MC_testing_labels = [float(l) for l in MC_testing_labels]

# batch_size = 35
# MC_train_loader = DataLoader(ClassificationDataset(MC_train_data, MC_training_labels, MC_seq_len), batch_size=batch_size, shuffle=True)
# MC_test_loader = DataLoader(ClassificationDataset(MC_test_data, MC_testing_labels, MC_seq_len), batch_size=batch_size)

# batch_size = 35
# MC_LM_train_loader = DataLoader(WordDataset(MC_train_data, MC_seq_len), batch_size=batch_size, shuffle=True, drop_last=True)
# MC_LM_test_loader = DataLoader(WordDataset(MC_test_data, MC_seq_len), batch_size=batch_size)

#################################################
####          Rel Pron (RP) Dataset          ####
#################################################

with open('./data/rp_train_data.txt') as f:
    RP_training_data_raw = f.readlines()

with open('./data/rp_test_data.txt') as f:
    RP_testing_data_raw = f.readlines()

vocab = dict()          # dictionary to be filled with the vocabulary in the form { word : POStag }
data = dict()           # dictionary to be filled with all the data (train, dev and test subsets); entries of the 
                        # form { sentence : label } with label encoding '1' as [1.0, 0.0] and '0' as [0.0, 1.0]
RP_training_data = []      # list of sentences in the train dataset as strings "word1 word2 ..."
RP_training_labels = []     # list of labels for the train dataset
RP_dev_data = []           # list of sentences in the dev dataset as strings "word1 word2 ..."
RP_dev_labels = []         # list of labels for the dev dataset
RP_testing_data = []       # list of sentences in the test dataset as strings "word1 word2 ..."
RP_testing_labels = []     # list of labels for the test dataset

# Go through the train data
for sent in RP_training_data_raw:
    words = sent[2:].split() 
    sent_untagged = ''
    for word in words:
        word_untagged, tag = word.split('_')
        vocab[word_untagged] = tag
        sent_untagged += word_untagged + ' '
    sentence = sent_untagged[:-1]
    RP_training_data.append(sentence)
    RP_training_labels.append(int(sent[0]))
    label = np.array([1.0, 0.0]) if sent[0] == '1' else np.array([0.0, 1.0])
    data[sentence] = label

for sent in RP_testing_data_raw:
    words = sent[2:].split() 
    sent_untagged = ''
    for word in words:
        word_untagged, tag = word.split('_')
        vocab[word_untagged] = tag
        sent_untagged += word_untagged + ' '
    sentence = sent_untagged[:-1]
    RP_testing_data.append(sentence)
    RP_testing_labels.append(int(sent[0]))
    label = np.array([1.0, 0.0]) if sent[0] == '1' else np.array([0.0, 1.0])
    data[sentence] = label

RP_seq_len = max(len(sent.split()) for sent in RP_training_data + RP_dev_data + RP_testing_data)

RP_tokenized_train = [s.lower().split() for s in RP_training_data]
RP_tokenized_test = [s.lower().split() for s in RP_testing_data]
RP_vocab = sorted(set(w for s in RP_tokenized_train + RP_tokenized_test for w in s))
RP_vocab = sorted(set(w for s in RP_tokenized_train for w in s))
RP_vocab = ["<pad>", "<unk>"] + RP_vocab
RP_word2idx = {w: i for i, w in enumerate(RP_vocab)}
RP_idx2word = {i: w for w, i in RP_word2idx.items()}

def encode_sentence(sent):
    return [RP_word2idx.get(w, RP_word2idx["<unk>"]) for w in sent]

RP_train_data = [encode_sentence(s) for s in RP_tokenized_train]
RP_test_data = [encode_sentence(s) for s in RP_tokenized_test]

RP_training_labels = [float(l) for l in RP_training_labels]
RP_testing_labels = [float(l) for l in RP_testing_labels]

# batch_size = 37
# RP_train_loader = DataLoader(ClassificationDataset(RP_train_data, RP_training_labels, RP_seq_len), batch_size=batch_size, shuffle=True, drop_last=True)
# RP_test_loader = DataLoader(ClassificationDataset(RP_test_data, RP_testing_labels, RP_seq_len), batch_size=batch_size)

# batch_size = 37
# unk_idx = RP_word2idx["<unk>"]
# unk_prob = 0.05
# RP_train_loader = DataLoader(ClassificationDataset(RP_train_data, RP_training_labels, RP_seq_len, unk_idx, unk_prob), batch_size=batch_size, shuffle=True, drop_last=True)
# RP_test_loader = DataLoader(ClassificationDataset(RP_test_data, RP_testing_labels, RP_seq_len), batch_size=batch_size)

def load_dataset(args):
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None
    if args.dataset == 'TS-LM' or args.dataset == 'TS':
        train_data, train_labels = LM_train_data, None
        test_data, test_labels = LM_test_data, None
        task = 'lm'
        seq_len = 6
        vocab_size = len(LM_vocab)
    elif args.dataset == 'MC':
        train_data, train_labels = MC_train_data, MC_training_labels
        test_data, test_labels = MC_test_data, MC_testing_labels
        task = 'binary'
        seq_len = MC_seq_len
        vocab_size = len(MC_vocab)
    elif args.dataset == 'MC-LM':
        train_data, train_labels = MC_train_data, None
        test_data, test_labels = MC_test_data, None
        task = 'lm'
        seq_len = MC_seq_len
        vocab_size = len(MC_vocab)
    elif args.dataset == 'RP':
        train_data, train_labels = RP_train_data, RP_training_labels
        test_data, test_labels = RP_test_data, RP_testing_labels
        task = 'binary'
        seq_len = RP_seq_len
        vocab_size = len(RP_vocab)
    else:
        raise ValueError("Unknown task. Choose from 'TS-LM', 'MC', 'MC-LM', or 'RP'.")

    batch_size = args.batch_size
    unk_idx = 1
    unk_prob = args.unk_prob if hasattr(args, 'unk_prob') else 0.0
    if hasattr(args, 'pad_seq_len') and args.pad_seq_len is not None:
        seq_len = args.pad_seq_len

    if task == 'lm':
        train_loader = DataLoader(WordDataset(train_data, seq_len), batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(WordDataset(test_data, seq_len), batch_size=batch_size)
    elif task == 'binary':
        train_loader = DataLoader(ClassificationDataset(train_data, train_labels, seq_len, unk_idx, unk_prob), batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ClassificationDataset(test_data, test_labels, seq_len), batch_size=batch_size)

    return train_loader, test_loader, seq_len, vocab_size, task