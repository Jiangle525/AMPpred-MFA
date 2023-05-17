import gc
import os
import argparse
from AMPpred_MFA.models.Model import BaseConfig
from AMPpred_MFA.lib.Data import divide_data_and_save, result_combination
from AMPpred_MFA.lib.Visualization import colorful_print, current_time
from training_and_testing import training_and_testing


def displace_and_training(file_path_pos, file_path_neg, split_ratio, model_name, feature_method, repeat_times, resample, save_dir):
    config = BaseConfig()
    result_rootpath = os.path.join(save_dir, '{}_times_{}_{}_{}'.format(
        repeat_times, model_name, feature_method, current_time()))
    os.makedirs(result_rootpath, exist_ok=True)

    for i in range(repeat_times):
        colorful_print(
            'data displace is runnig at {} th...'.format(i+1), 'cyan', 'default')
        i_trial = '{}_trial'.format(i+1)
        result_save_dir = os.path.join(result_rootpath, i_trial)
        os.makedirs(result_save_dir, exist_ok=True)

        if resample:
            data_save_dir = os.path.join(result_rootpath,'data',i_trial)
            os.makedirs(data_save_dir, exist_ok=True)
            trainset_path = os.path.join(data_save_dir, config.save_trainset)
            testset_path = os.path.join(data_save_dir, config.save_testset)
            divide_data_and_save(file_path_pos, file_path_neg,
                                 trainset_path, testset_path, split_ratio=split_ratio)
            if split_ratio==0:
                testset_path=''
        else:
            data_dir = os.path.join('data',i_trial)
            trainset_path = os.path.join(data_dir, config.save_trainset)
            testset_path = os.path.join(data_dir, config.save_testset)
        training_and_testing(trainset_path, testset_path,
                             model_name, feature_method, result_save_dir)
        gc.collect()
    result_combination(result_rootpath, config.save_testing_log)


def get_args():
    parser = argparse.ArgumentParser(description='Displace data and training')
    parser.add_argument('--pos', type=str, default='./dataset/our_dataset/amps.fasta',
                        help='Choose positive dataset path')
    parser.add_argument('--neg', type=str, default='./dataset/our_dataset/non_amps.fasta',
                        help='Choose negative dataset path')
    parser.add_argument('--ratio', type=float, default=0.15,
                        help='Test set partition ratio (default: 0.15)')
    parser.add_argument('--model', type=str, required=True,
                        help='Choose model (case-sensitive): AMPpred_MFA, Att_BiLSTM, Att_ConvNet, BiLSTM, ConvNet')
    parser.add_argument('--feature', type=str, required=True,
                        help='Chosss feature method (case-insensitive): DDE, Vocab, Mixed')
    parser.add_argument('--times', type=int, default=5,
                        help='Numbers of displace times (default: 5)')
    parser.add_argument('--resample', type=bool, default=False,
                        help='Resampling and generating the dataset (default: False)')
    parser.add_argument('--save', type=str, default='result',
                        help='Result save path (default: result)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    pos = args.pos
    neg = args.neg
    split_ratio = args.ratio
    model_name = args.model
    feature_method = args.feature.lower()
    repeat_times = args.times
    resample = args.resample
    save_dir = args.save
    displace_and_training(pos, neg, split_ratio, model_name,
                          feature_method, repeat_times, resample, save_dir)
