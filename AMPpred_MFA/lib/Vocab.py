import collections
import json
from .Visualization import colorful_print


class Vocab:
    """ Vocabulary for biological sequence. """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(),
                                   key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def count_corpus(self, tokens):
        """ Count token frequencies. """
        # Here `tokens` is a 1D list or 2D list
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # Flatten a list of token lists into a list of tokens
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def tokenize(seqs, k_mer):
    return [[seq[i:i + k_mer] for i in range(len(seq) - k_mer + 1)]for seq in seqs]


def build_vocab(seqs, k_mer, min_freq=1, reserved_tokens=None):
    colorful_print('Building vocabulary...', 'cyan', 'default')
    tokens = tokenize(seqs, k_mer)
    vocab = Vocab(tokens, min_freq, reserved_tokens=reserved_tokens)
    colorful_print(
        'Vocabulary has beed built.', 'green', 'bold')
    return vocab


def save_vocab(vocab: Vocab, file_path):
    with open(file_path, 'w') as f:
        json.dump({
            'idx_to_token': vocab.idx_to_token,
            'token_to_idx': vocab.token_to_idx,
            '_token_freqs': vocab._token_freqs
        }, f)
    colorful_print(
        'Vocabulary is saved in: ' + file_path, 'green', 'bold')


def load_vocab(file_path):
    colorful_print('Loading vocabulary...', 'cyan', 'default')
    with open(file_path, 'r') as f:
        data = json.load(f)
    vocab = Vocab(tokens=[])
    vocab.idx_to_token = data['idx_to_token']
    vocab.token_to_idx = data['token_to_idx']
    vocab._token_freqs = data['_token_freqs']
    colorful_print(
        'Vocabulary has been loaded.', 'green', 'bold')
    return vocab
