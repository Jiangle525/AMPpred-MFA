import os
import torch
import numpy as np
import pandas as pd
from .Visualization import colorful


def load_fasta_from_str(fastas, sep='\n'):
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    lst_fastas = []
    try:
        lines = fastas.strip().split(sep)
        n_lines = len(lines)
        i = 0
        while i < n_lines:
            if lines[i] and lines[i][0] == '>':
                name = lines[i].strip()
                i += 1
                sequence = ''
                while i < n_lines and lines[i] and lines[i][0] != '>':
                    sequence += lines[i].strip()
                    i += 1
                assert set(sequence).issubset(
                    standard_aa), 'The sequence must be composed of standard amino acids'
                lst_fastas.append((name, sequence))
            else:
                i += 1
    except Exception:
        raise ValueError('Fasta file format error or inaccessible!')
    return np.array(lst_fastas)


def load_fasta_from_file(file_path):
    return load_fasta_from_str(open(file_path, 'r', encoding='utf-8').read())


def save_fasta_with_flag(lst_fastas, labels, file_path, flag='training'):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['{}|{}|{}\n{}'.format(fasta[0],
                                                 label, flag, fasta[1]) for fasta, label in zip(lst_fastas, labels)]) + '\n')


def undersample(pos, neg, undersample_ratio=1, shuffle=False):
    choice_num = int(len(pos)*undersample_ratio)
    if choice_num <= len(neg):
        idx = torch.randperm(len(neg))[:choice_num]
        neg = np.array([neg[i] for i in idx])
    dataset = np.r_[pos, neg]
    labels = np.r_[np.ones(len(pos), dtype=int),
                   np.zeros(len(neg), dtype=int)]
    if shuffle:
        shuffle_dataset(dataset, labels)
    return dataset, labels


def shuffle_dataset(x, y):
    random_seed = np.random.randint(0, 1234)
    np.random.seed(random_seed)
    np.random.shuffle(x)
    np.random.seed(random_seed)
    np.random.shuffle(y)


def divide_data_and_save(file_path_pos, file_path_neg, save_path1, save_path2, split_ratio=0):
    data_pos = load_fasta_from_file(file_path_pos)
    data_neg = load_fasta_from_file(file_path_neg)
    dataset, labels = undersample(data_pos, data_neg, undersample_ratio=1, shuffle=True)
    num_split = int(dataset.shape[0] * split_ratio)
    save_fasta_with_flag(dataset[num_split:], labels[num_split:],
                         save_path1, 'training')
    if num_split:
        save_fasta_with_flag(dataset[:num_split], labels[:num_split],
                             save_path2, 'testing')


def load_dataset(file_path):
    dataset = []
    lst_fastas = load_fasta_from_file(file_path)
    for fasta in lst_fastas:
        assert '|' in fasta[0], 'Every sequence\'s header must be ">name|label|flag", such as >Proten Name|0|training'
        lst_name = fasta[0].split('|')
        # The name can contain multiple '|'
        name = '|'.join(lst_name[:-2])
        label = lst_name[-2]
        dataset.append((name, fasta[1], label))

    return np.array(dataset)


def get_dataset_labels(dataset):
    return torch.tensor(dataset[:, -1].astype(np.int64), dtype=torch.int64)


def get_data_tensor_loader(dataset, batch_size, shuffle):
    dataset = torch.utils.data.TensorDataset(*dataset)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)


def get_data_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)


def get_data_iter(tuple_dataset, batch_size=64, shuffle=False, split_ratio=0):
    num_total = tuple_dataset[0].shape[0]
    num_split = int(np.ceil(num_total * split_ratio))
    tuple_data = (i[num_split:] for i in tuple_dataset)
    data_iter = get_data_tensor_loader(tuple_data, batch_size, shuffle=shuffle)
    if num_split:
        tuple_split = (i[:num_split] for i in tuple_dataset)
        split_iter = get_data_tensor_loader(
            tuple_split, batch_size, shuffle=shuffle)
        return data_iter, split_iter
    return data_iter


def iter_info(iter_data):
    print(colorful('Batch numbers:', 'white', 'bold'), colorful(
        len(iter_data), 'cyan', 'default'), end=' ' * 4)
    for lst in iter_data:
        print(colorful('Batch size:', 'white', 'bold'),
              colorful(lst[0].shape[0], 'cyan', 'default'))
        for i in range(len(lst) - 1):
            print(colorful('X{}.shape:'.format("" if len(lst) == 2 else i+1), 'white', 'bold'), colorful(
                lst[i].shape, 'cyan', 'default'), end=' ' * 4)
        print(colorful('y.shape:', 'white', 'bold'),
              colorful(lst[-1].shape, 'cyan', 'default'))
        break


def result_combination(root_dir, sub_dir):
    df = pd.DataFrame()
    for dir_name in os.listdir(root_dir):
        if 'data'==dir_name:
            continue
        if os.path.isdir(os.path.join(root_dir, dir_name)):
            result_dir = os.path.join(root_dir, dir_name, sub_dir)
            for file_name in os.listdir(result_dir):
                if '.csv' in file_name:
                    df = df._append(pd.read_csv(
                        os.path.join(result_dir, file_name)))
    df_mean = []
    for col in df:
        if col == 'CM':
            df_mean.append(np.array([eval(i) for i in df[col].tolist()]).mean(
                axis=0).astype(int).tolist())
        else:
            df_mean.append(str(round(df[col].mean(), 3)) + '±' + 
                           str(round(df[col].std(), 3)))
    df.insert(0, 'Trial Number', ['{} th'.format(i+1)
              for i in range(len(df))], allow_duplicates=False)
    df.loc[len(df)] = ['Mean']+df_mean
    df.to_csv(os.path.join(root_dir, 'result.csv'), index=False)
    print('result has combinated in:', os.path.join(root_dir, 'result.csv'))
