import torch
import numpy as np
from . import Vocab

def easy_encoding(dataset, encoding_method, vocab=None, k_mer=1, padding_size=100):
    """dataset.shape: (n, 2), 
    第一个维度表示序列;
    第二个维度的第一个元素为fasta的name,
    第二个维度的第二个元素为fasta的sequence
    
    返回一个包含编码的元组
    """
    encoding_method = encoding_method.lower()
    if encoding_method == 'dde':
        x_encodings = encoding_by_dde(dataset[:,1]),
    if encoding_method == 'vocab':
        x_encodings = encoding_by_vocab(vocab, dataset[:,1], k_mer, padding_size),
    if encoding_method=='mixed':
        x_encodings = encoding_by_dde(dataset[:,1]),encoding_by_vocab(
            vocab, dataset[:,1],k_mer,padding_size)
    return x_encodings

def truncate_pad(line, num_steps, padding_token):
    """ Truncate or pad sequences. """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def encoding_by_vocab(vocab, seqs, k_mer, padding_size):
    tokens = Vocab.tokenize(seqs, k_mer)
    return torch.tensor([
        truncate_pad(vocab[token], padding_size, vocab['<pad>'])
        for token in tokens
    ])


def encoding_by_dde(seqs):
    return torch.tensor(DDE(seqs), dtype=torch.float32)


def DDE(seqs):
    codons_table = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2,
                    'L': 6, 'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2}
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    C_N = 61
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    all_DDE_p = []
    T_m = []

    # Calculate T_m
    for pair in diPeptides:
        T_m.append(
            (codons_table[pair[0]] / C_N) * (codons_table[pair[1]] / C_N))

    for seq in seqs:
        N = len(seq) - 1
        D_c = []
        T_v = []
        DDE_p = []

        # Calculate D_c
        for i in range(len(diPeptides)):
            D_c.append(seq.count(diPeptides[i]) / N)

        # Calculate T_v
        for i in range(len(diPeptides)):
            T_v.append(T_m[i] * (1 - T_m[i]) / N)

        # Calculate DDP_p
        for i in range(len(diPeptides)):
            DDE_p.append((D_c[i] - T_m[i]) / np.sqrt(T_v[i]))

        all_DDE_p.append(DDE_p)
    return np.array(all_DDE_p)


