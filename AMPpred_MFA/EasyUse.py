import torch
import pandas as pd
from .lib.Vocab import load_vocab
from .lib.Encoding import easy_encoding
from .lib.Data import get_data_iter, get_data_loader, load_dataset
from .models.Model import predict, load_model
from .models.AMPpred_MFA import Model, Config


def easy_predict(fastas, model_path, vocab_path):
    """Easy prediction for mixed model(AMPpred_MFA) from fastas

    Args:
        fastas (_type_): fastas.shape: (n, 2), fastas[:,0] represents names, fastas[:,1] represents sequences
        model_path (_type_): AMPpred_MFA model path
        vocab_path (_type_): vocab path

    Returns:
        _type_: result in pandas.DataFrame
    """
    # load vocab and configure config of AMPpredMFA model
    vocab = load_vocab(vocab_path)
    config = Config()
    config.batch_size = 32
    config.embed_padding_idx = vocab[config.padding_token]
    config.feature_dim = 400
    config.vocab_size = len(vocab)
    # load model of AMPpred_MFA
    model = Model(config)
    load_model(model, model_path)
    data_x = easy_encoding(fastas, 'mixed', vocab,
                           config.k_mer, config.padding_size)
    data_iter = get_data_iter(data_x, config.batch_size,
                              shuffle=False, split_ratio=0)
    seqs_iter = get_data_loader(
        fastas[:, 1], config.batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = predict(model, data_iter, seqs_iter, device)
    names_sequences = ['\r\n'.join(i) for i in fastas]
    df_result = pd.DataFrame(result).transpose()
    df_result = pd.concat(
        [pd.DataFrame(names_sequences), df_result], axis=1)
    df_result.columns = ['Sequence', 'Class', 'Probability', 'Motif']
    df_result['Class'] = df_result['Class'].apply(
        lambda x: 'AMPs'if x == 1 else 'Non-AMPs')
    return df_result


def easy_predict_from_file(fasta_file, model_path, vocab_path):
    """Easy prediction for mixed model(AMPpred_MFA) from fasta file

    Args:
        fasta_file (_type_): fasta file path
        model_path (_type_): AMPpred_MFA model path
        vocab_path (_type_): vocab path

    Returns:
        _type_: result in pandas.DataFrame
    """
    fastas = load_dataset(fasta_file)
    return easy_predict(fastas, model_path, vocab_path)
