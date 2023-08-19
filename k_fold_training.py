import gc
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from AMPpred_MFA.models.Model import BaseConfig
from AMPpred_MFA.lib.Data import load_fasta_from_file, result_combination, save_fasta_with_flag, undersample
from AMPpred_MFA.lib.Visualization import colorful_print, current_time, draw_roc
from training_and_testing import training_and_testing



def split_k_fold(file_path_pos, file_path_neg, save_path_dir, k_fold=10):
    os.makedirs(save_path_dir, exist_ok=True)
    config = BaseConfig()
    data_pos = load_fasta_from_file(file_path_pos)
    data_neg = load_fasta_from_file(file_path_neg)
    dataset, labels = undersample(data_pos, data_neg, undersample_ratio=1, shuffle=True)
    kf = KFold(n_splits=k_fold, shuffle=True)  
    i = 1
    for train_index, test_index in kf.split(dataset):
        i_fold = '{}_fold'.format(i)
        data_save_dir = os.path.join(save_path_dir, 'data', i_fold)
        os.makedirs(data_save_dir, exist_ok=True)
        trainset_path = os.path.join(data_save_dir, config.save_trainset)
        testset_path = os.path.join(data_save_dir, config.save_testset)

        x_train, y_train = dataset[train_index], labels[train_index]
        x_test, y_test = dataset[test_index], labels[test_index]
        save_fasta_with_flag(x_train, y_train,
                         trainset_path, 'training')
        save_fasta_with_flag(x_test, y_test,
                        testset_path, 'testing')
        colorful_print('{} fold has been created.'.format(i), 'cyan', 'default')
        i+=1

def get_rocs(root_dir):
    fprs, tprs = [], []

    for sub_dir in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, sub_dir)) or sub_dir=='data':
            continue
        roc_dir = os.path.join(root_dir, sub_dir, 'testing log')
        for roc_file in os.listdir(roc_dir):
            if 'roc' not in roc_file:
                continue
            roc_path = os.path.join(roc_dir, roc_file)
            method_name = roc_file.split('_')[-1][:-4]
            method_name = sub_dir if method_name == 'roc' else method_name
            roc_df = pd.read_csv(roc_path)
            fprs.append(roc_df['fpr'])
            tprs.append(roc_df['tpr'])
    avg_auc = pd.read_csv(os.path.join(root_dir,'result.csv'))['AUC'].iloc[-1].split('±')
    avg_auc = list(map(float,avg_auc))
    return fprs, tprs, avg_auc


def k_fold_training(file_path_pos, file_path_neg, model_name, feature_method, k_fold, save_dir):
    config = BaseConfig()
    result_rootpath = os.path.join(save_dir, '{}_fold_{}_{}_{}'.format(
        k_fold, model_name, feature_method, current_time()))
    os.makedirs(result_rootpath, exist_ok=True)
    split_k_fold(file_path_pos, file_path_neg, result_rootpath, 10)

    for i in range(k_fold):
        colorful_print(
            'The {} fold is runnig at {} th...'.format(k_fold, i+1), 'cyan', 'default')
        i_fold = '{}_fold'.format(i+1)
        result_save_dir = os.path.join(result_rootpath, i_fold)
        os.makedirs(result_save_dir, exist_ok=True)
        trainset_path = os.path.join(result_rootpath, 'data', i_fold, config.save_trainset)
        testset_path = os.path.join(result_rootpath, 'data', i_fold, config.save_testset)
        training_and_testing(trainset_path, testset_path,
                             model_name, feature_method, result_save_dir)
        gc.collect()
    result_combination(result_rootpath, config.save_testing_log)
    fig = draw_roc(*get_rocs(result_rootpath))
    fig.savefig(os.path.join(result_rootpath,'roc.png'), dpi=300)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description='Displace data and training')
    parser.add_argument('--pos', type=str, default='./dataset/our_dataset/amps.fasta',
                        help='Choose positive dataset path')
    parser.add_argument('--neg', type=str, default='./dataset/our_dataset/non_amps.fasta',
                        help='Choose negative dataset path')
    parser.add_argument('--model', type=str, required=True,
                        help='Choose model (case-sensitive): AMPpred_MFA, Att_BiLSTM, Att_ConvNet, BiLSTM, ConvNet')
    parser.add_argument('--feature', type=str, required=True,
                        help='Chosss feature method (case-insensitive): DDE, Vocab, Mixed')
    parser.add_argument('--fold', type=int, default=10,
                        help='Numbers of k fold (default: 10)')
    parser.add_argument('--save', type=str, default='result',
                        help='Result save path (default: result)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    pos = args.pos
    neg = args.neg
    model_name = args.model
    feature_method = args.feature.lower()
    k_fold = args.fold
    save_dir = args.save
    k_fold_training(pos, neg, model_name, feature_method, k_fold, save_dir)
