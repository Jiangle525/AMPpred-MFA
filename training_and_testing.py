import os
import argparse
from importlib import import_module
from AMPpred_MFA.lib.Data import load_dataset, get_dataset_labels, iter_info, get_data_iter
from AMPpred_MFA.lib.Encoding import easy_encoding
from AMPpred_MFA.lib.Vocab import build_vocab, load_vocab, save_vocab
from AMPpred_MFA.models.Model import BaseConfig, train, test, save_model
from AMPpred_MFA.lib.Visualization import colorful_print


def training_and_testing(trainset_path, testset_path, model_name, feature_method, save_dir, vocab_path=None):
    """  实例化配置对象、构建词表、生成迭代器  """
    feature_method = feature_method.lower()
    if feature_method == 'mixed':
        assert 'amppred_mfa' == model_name.lower(
        ), 'Mixed features must match the AMPpred_MFA!'
    model_file = import_module('AMPpred_MFA.models.' + model_name)
    config: BaseConfig = model_file.Config()
    os.makedirs(os.path.join(
        save_dir, config.save_training_log), exist_ok=True)
    os.makedirs(os.path.join(save_dir, config.save_testing_log), exist_ok=True)

    vocab = None
    if feature_method in ['vocab', 'mixed']:
        if vocab_path:
            vocab = load_vocab(vocab_path)
        else:
            trainset = load_dataset(trainset_path)
            vocab = build_vocab(trainset[:, 1], k_mer=config.k_mer,
                                reserved_tokens=[config.padding_token])
            vocab_save_path = os.path.join(save_dir, config.save_vocab)
            save_vocab(vocab, vocab_save_path)
        config.vocab_size = len(vocab)
        config.embed_padding_idx = vocab[config.padding_token]

    train_dataset = load_dataset(trainset_path)
    train_x = easy_encoding(train_dataset, feature_method, vocab,
                            config.k_mer, config.padding_size)
    train_y = get_dataset_labels(train_dataset)
    train_iter, valid_iter = get_data_iter((*train_x, train_y), config.batch_size,
                                           shuffle=True, split_ratio=config.train_valid_ratio)

    colorful_print('\nTrain set iterator info:', 'white', 'bold')
    iter_info(train_iter)
    colorful_print('\nValidation set iterator info:', 'white', 'bold')
    iter_info(valid_iter)
    print()

    # 模型设置
    config.feature_method = feature_method
    if feature_method in ['dde', 'mixed']:
        config.feature_dim = list(train_iter.dataset)[0][0].shape[-1]
    # list(train_iter.dataset)为列表，可能包含多个特征；
    # list(train_iter.dataset)[0][0]为第0个特征的X，list(train_iter.dataset)[0][1]为第0个特征的y

    """  创建模型  """
    model = model_file.Model(config)
    # print(model)
    # for x,y in train_iter:
    #     summary(model, x)
    #     break

    """  训练模型、保存训练过程  """
    model = train(config, model, train_iter, valid_iter, save_dir)
    save_model(model, os.path.join(save_dir, config.save_model))

    """  测试模型  """
    if testset_path:
        test_dataset = load_dataset(testset_path)
        test_x = easy_encoding(test_dataset, feature_method, vocab,
                               config.k_mer, config.padding_size)
        test_y = get_dataset_labels(test_dataset)
        test_iter = get_data_iter(
            (*test_x, test_y), config.batch_size, shuffle=False)
        test(config, model, test_iter, True, save_dir)


def get_args():
    parser = argparse.ArgumentParser(description='Sequences Classification')
    parser.add_argument('--trainset', type=str, required=True,
                        help='Choose train set path')
    parser.add_argument('--testset', type=str, default='',
                        help='Choose test set path')
    parser.add_argument('--model', type=str, required=True,
                        help='Choose model: AMPpred_MFA, Att_BiLSTM, Att_ConvNet, BiLSTM, ConvNet')
    parser.add_argument('--feature', type=str, required=True,
                        help='Chosss feature method (case-insensitive): DDE, Vocab, Mixed')
    parser.add_argument('--save', type=str, required=True,
                        help='Choose result save directory')
    parser.add_argument('--vocab', type=str, default='',
                        help='Choose vocabulary path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    file_path = args.trainset
    testset_path = args.testset
    model_name = args.model
    feature_method = args.feature
    save_dir = args.save
    vocab_path = args.vocab
    training_and_testing(file_path, testset_path, model_name,
                         feature_method, save_dir, vocab_path)
