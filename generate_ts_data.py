import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import math

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

import random

# remove all articles (the, a, an)
# Vocabulary
subjects = ["man", "woman", "boy", "girl"]
verbs = ["eats", "sees", "likes", "finds"]
adjectives = ["big", "small", "yellow", "green"]
objects = ["dog", "cat", "apple", "banana"]
prepositions = ["on", "under", "from"]
places = ["mat", "table", "floor"]

# HARD CONSTRAINTS
disallowed_adj_obj = {
    ("green", "dog"),
    ("green", "cat")
}

disallowed_verb_obj = {
    ("eats", "dog"),
    ("eats", "cat")
}

disallowed_prep_combos = {
    ("under", "floor"),
    ("from", "floor"),
}

disallowed_verb_prep = {
    ("likes", "from"),
    ("finds", "from")
}

# SOFT PROBABILITIES
adjective_probs = {
    "banana": {"yellow": 0.4, "green": 0.1, "big": 0.25, "small": 0.25},
    "apple": {"yellow": 0.4, "green": 0.4, "big": 0.1, "small": 0.1},
    "dog": {"yellow": 0.1, "big": 0.45, "small": 0.45},
    "cat": {"yellow": 0.1, "big": 0.45, "small": 0.45}
}

verb_object_probs = {
    ("man", "eats"): {"banana": 0.7, "apple": 0.3},
    ("girl", "likes"): {"cat": 0.7, "dog": 0.3},
    ("woman", "finds"): {"apple": 0.7, "banana": 0.3},
    ("boy", "sees"): {"dog": 0.7, "cat": 0.3},
}

def weighted_choice(d):
    items = list(d.items())
    choices, weights = zip(*items)
    return random.choices(choices, weights=weights, k=1)[0]

def generate_sentence(max_attempts=50):
    for _ in range(max_attempts):
        subj = random.choice(subjects)
        verb = random.choice(verbs)

        # Pick object with biased sampling if possible
        obj_choices = verb_object_probs.get((subj, verb))
        if obj_choices:
            obj = weighted_choice(obj_choices)
        else:
            obj = random.choice(objects)
        
        # Pick adjective if applicable
        adj = None
        if obj in adjective_probs:
            adj_probs = adjective_probs[obj]
            adj = weighted_choice(adj_probs)
            if (adj, obj) in disallowed_adj_obj:
                adj = None
        else:
            # No adjective
            adj = None
        
        # Verb-object constraints
        if (verb, obj) in disallowed_verb_obj:
            continue
        
        # Compose full object phrase
        obj_phrase = f"{adj} {obj}" if adj else obj

        # Add preposition/place?
        use_prep = random.random() < 0.5
        prep = random.choice(prepositions) if use_prep else None
        place = random.choice(places) if prep else None

        # Preposition constraints
        if prep and (prep, place) in disallowed_prep_combos:
            continue
        if prep and (verb, prep) in disallowed_verb_prep:
            continue

        # Build structure
        sentence = f"{subj} {verb} {obj_phrase}"
        if prep and place:
            sentence += f" {prep} {place}"

        return sentence

    return "boy sees dog"  # fallback

# Step 1: Generate training set
train_sentences = []
train_sentences_set = set()
while len(train_sentences_set) < 200:
    sent = generate_sentence()
    train_sentences.append(sent)
    train_sentences_set.add(sent)

print(len(train_sentences_set), "unique training sentences generated.")
train_sentences = list(train_sentences_set)

# Step 2: Generate test set with no overlap
test_sentences = set()
while len(test_sentences) < 50:
    sent = generate_sentence()
    if sent not in train_sentences_set:
        test_sentences.add(sent)

# Optionally convert to lists
# train_sentences = list(train_sentences)
test_sentences = list(test_sentences)

print(len(test_sentences), "unique test sentences generated.")

tokenized_train = [s.lower().split() for s in train_sentences]
tokenized_test = [s.lower().split() for s in test_sentences]
vocab = sorted(set(w for s in tokenized_train for w in s))
vocab = ["<pad>", "<unk>"] + vocab
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

def encode_sentence(sent):
    return [word2idx.get(w, word2idx["<unk>"]) for w in sent]

train_data = [encode_sentence(s) for s in tokenized_train]
test_data = [encode_sentence(s) for s in tokenized_test]
random.shuffle(train_data)

LM_vocab = vocab
seq_len = LM_seq_len = 6
batch_size = 32
train_loader = DataLoader(WordDataset(train_data, seq_len), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(WordDataset(test_data, seq_len), batch_size=batch_size)

with open("./datasets/TS_LM_data_train.txt", "w") as f:
    for sent in train_sentences:
        f.write(sent + "\n")

with open("./datasets/TS_LM_data_test.txt", "w") as f:
    for sent in test_sentences:
        f.write(sent + "\n")