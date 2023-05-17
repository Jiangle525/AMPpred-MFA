import os
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from ..lib.Visualization import *


class BaseConfig:
    def __init__(self, vocab_size=0, embed_padding_idx=0):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # input settings
        self.k_mer = 1                                  # k mer size
        self.padding_size = 101-self.k_mer              # truncation padding size
        self.padding_token = '<padding>'                # embedding token
        self.train_valid_ratio = 0.2                    # train valid division ratio
        # input of model: DDE, vocab, mixed
        self.feature_method = 'dde'
        self.vocab_size = vocab_size                    # vocabulary size
        self.embed_padding_idx = embed_padding_idx      # embedding padding token's id
        self.num_classes = 2                            # numbers of classes
        self.feature_dim = 400                          # feature dimension
        self.embedding_dim = 60                         # embedding dimension

        # result settings
        self.save_training_log = 'training log'         # training log save path
        self.save_testing_log = 'testing log'           # testing log save path
        self.save_trainset = 'train.fasta'              # train set save path
        self.save_testset = 'test.fasta'                # test set save path
        self.save_roc = "roc.txt"                       # tpr and frp save path
        self.save_model = 'model.pth'                   # model save path
        self.save_vocab = 'vocab.json'                  # vocabulary save path
        self.save_feaures = 'features.txt'              # features save path
        self.print_logpath = 'print.log'                # print log save path
        self.jpg_dpi = 300                              # save jpg dpi


def save_model(model, save_path):
    colorful_print('Saving model...', 'cyan', 'default')
    torch.save(model.state_dict(), save_path)
    colorful_print(
        'Model is saved in: ' + save_path, 'green', 'bold')


def load_model(model, model_path):
    colorful_print('Loading model...', 'cyan', 'default')
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    colorful_print(
        'Model has been loaded.', 'green', 'bold')


def evaluate(model, data_iter, device, loss_func=None, test_flag=False):
    """evaluating model on data_iter

    Args:
        model (_type_): pytorch model
        data_iter (_type_): data iterator
        device (_type_): device of eavluation
        loss_func (_type_, optional): loss function. Defaults to None.
        test_flag (bool, optional): test falg. Defaults to False.

    Returns:
        _type_: metrics of evaluation
    """
    model.eval()
    loss_total = 0
    y_true_all, y_pred_all, y_pred_proba_all = [], [], []

    with torch.no_grad():
        for dataset in data_iter:
            # dataset为list,前len(dataset)-1个元素为输入, 最后一个元素为输出
            dataset = [dataset[i].to(device) for i in range(len(dataset))]
            outputs = model(dataset[:-1])
            if not test_flag:
                loss = loss_func(outputs, dataset[-1])
                loss_total += loss.item()
            y_true = dataset[-1].data.cpu().numpy()
            y_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            y_pred_proba = outputs.data.cpu().numpy()[:, 1]
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            y_pred_proba_all.extend(y_pred_proba)
    if not test_flag:
        acc = metrics.accuracy_score(y_true_all, y_pred_all)
        return loss_total / len(data_iter), acc
    # this is test return
    cm = metrics.confusion_matrix(y_true_all, y_pred_all)
    if len(cm) > 2:
        return cm
    auc = metrics.roc_auc_score(y_true_all, y_pred_proba_all)
    fpr, tpr, _ = metrics.roc_curve(y_true_all, y_pred_proba_all)
    return cm, auc, fpr, tpr


def train(config: BaseConfig, model, train_iter, valid_iter=None, save_dir="./"):
    """training model on train_iter and validating on valid_iter

    Args:
        config (BaseConfig): config of Model
        model (_type_): pytorch model
        train_iter (_type_): training iterator
        valid_iter (_type_, optional): validation iterator. Defaults to None.
        save_dir (str, optional): result save path. Defaults to "./".

    Returns:
        _type_: trained model
    """
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    epoch_iterations = 0  # 记录迭代次数
    patience_counter = 0  # 忍耐器计数器
    last_valid_loss = float("inf")  # 验证集上一次损失值
    num_batchs = len(train_iter)  # 批量数
    training_savepath = os.path.join(
        save_dir, config.save_training_log, current_time() + ".jpg")
    evaluation_metrics = {"train loss": [], "valid loss": [],
                          "train acc": [], "valid acc": []}
    evaluation_metrics["train loss"].append(last_valid_loss)
    evaluation_metrics["valid loss"].append(last_valid_loss)
    evaluation_metrics["train acc"].append(0.5)
    evaluation_metrics["valid acc"].append(0.5)

    # 控制训练输出信息的格式化
    train_log_wait_print = ", ".join(list(filter(None, ["Batch [{}/{}]".format(
        colorful("{:0>3}", "cyan", "default"), colorful("{:0>3}", "cyan", "default"),),
        "train loss: {}".format(
            colorful("{:->5}".format(""), "cyan", "blink")),
        "train acc: {}".format(colorful("{:->5}".format(""), "cyan", "blink")),
        "valid loss: {}".format(
            colorful("{:->5}".format(""), "cyan", "blink")) if valid_iter else "",
        "valid acc: {}".format(
            colorful("{:->5}".format(""), "cyan", "blink")) if valid_iter else ""
    ]
    )))
    train_log_value_print = ", ".join(list(filter(None, ["Batch [{}/{}]".format(
        colorful("{:0>3}", "cyan", "default"), colorful("{:0>3}", "cyan", "default"),),
        "train loss: {}".format(colorful("{:.3f}", "cyan", "default")),
        "train acc: {}".format(colorful("{:.3f}", "cyan", "default")),
        "valid loss: {}".format(
            colorful("{:.3f}", "cyan", "default")) if valid_iter else "",
        "valid acc: {}".format(
            colorful("{:.3f}", "cyan", "default")) if valid_iter else ""
    ]
    )))

    # 迭代训练
    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]:".format(
            colorful(epoch + 1, "cyan", "default"),
            colorful(config.num_epochs, "cyan", "default"),
        ))
        model.train()
        epoch_iterations += 1
        train_loss_total, train_true_all, train_pred_all = 0, [], []

        for batch, dataset in enumerate(train_iter, 1):
            # dataset为list,前len(dataset)-1个元素为输入, 最后一个元素为输出
            dataset = [dataset[i].to(config.device)
                       for i in range(len(dataset))]
            outputs = model(dataset[:-1])
            optimizer.zero_grad()
            loss = loss_func(outputs, dataset[-1])
            train_loss_total += loss.item()
            y_true = dataset[-1].data.cpu().numpy()
            y_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            train_true_all.extend(y_true)
            train_pred_all.extend(y_pred)
            loss.backward()
            optimizer.step()
            cover_print(train_log_wait_print.format(
                batch, num_batchs), newline=False)
        train_loss = train_loss_total / len(train_iter)
        train_acc = metrics.accuracy_score(
            train_true_all, train_pred_all)
        evaluation_metrics["train loss"].append(train_loss)
        evaluation_metrics["train acc"].append(train_acc)
        if valid_iter:
            valid_loss, valid_acc = evaluate(
                model, valid_iter, config.device, loss_func)
            cover_print(train_log_value_print.format(
                batch, num_batchs, train_loss, train_acc, valid_loss, valid_acc), newline=True)
            evaluation_metrics["valid loss"].append(valid_loss)
            evaluation_metrics["valid acc"].append(valid_acc)
            # 早停法
            if valid_loss < last_valid_loss:
                patience_counter = 0  # 忍耐器计数器归零
                last_valid_loss = valid_loss
            else:
                patience_counter += 1
                if patience_counter >= config.num_patience:
                    print("The model has not been improved for {} epochs, training over!".format(
                        patience_counter))
                    break
        else:
            cover_print(train_log_value_print.format(
                batch, num_batchs, train_loss, train_acc), newline=True,)
    draw_metrics(evaluation_metrics).savefig(
        training_savepath, dpi=config.jpg_dpi)
    colorful_print("Training processing is saved in: " +
                   training_savepath, "green", "bold")
    return model


def test(config: BaseConfig, model, test_iter, is_cover=True, save_dir="./"):
    """testing on test_iter

    Args:
        config (BaseConfig): config of Model
        model (_type_): pytorch model
        test_iter (_type_): testing iterator
        is_cover (bool, optional): cover result file or not. Defaults to True.
        save_dir (str, optional): result save path. Defaults to "./".
    """
    model.eval()
    result = evaluate(model, test_iter, config.device, test_flag=True)
    cr = {}
    if len(result) == 1:
        cm = result
    else:
        cm, auc, fpr, tpr = result
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    cr["AUC"] = auc
    cr["Acc"] = (TN + TP) / (TN + FP + FN + TP)
    cr["MCC"] = ((TP * TN) - (FP * FN)) / \
        (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    cr["Sn"] = TP / (TP + FN)
    cr["Sp"] = TN / (TN + FP)
    cr['CM'] = str(cm.tolist())
    # get save path
    test_roc_fig_savepath = os.path.join(
        save_dir, config.save_testing_log, current_time() + ".jpg")
    test_roc_curve_savepath = os.path.join(
        save_dir, config.save_testing_log, config.save_roc)
    test_metrics_savepath = os.path.join(
        save_dir, config.save_testing_log, current_time() + ".csv")
    # save result
    draw_roc([fpr], [tpr]).savefig(test_roc_fig_savepath, dpi=config.jpg_dpi)
    roc_df = pd.DataFrame((fpr, tpr), index=['fpr', 'tpr']).T
    roc_df.to_csv(test_roc_curve_savepath, index=False)
    if is_cover:
        pd.DataFrame([cr]).to_csv(test_metrics_savepath, index=None, sep=",")
    else:
        pd.DataFrame([cr]).to_csv(test_metrics_savepath,
                                  header=None, mode="a", index=None, sep=",")
    colorful_print("Testing roc curve is saved in: " +
                   test_roc_fig_savepath, "green", "bold")
    colorful_print("Testing fpr and tpr is saved in: " +
                   test_roc_curve_savepath, "green", "bold")
    colorful_print("Tesing metrics is saved in: " +
                   test_metrics_savepath, "green", "bold")


def predict(model, data_iter, seqs_iter, device='cpu'):
    """predicting data_iter on model and inferring motif

    Args:
        model (_type_): pytorch model
        data_iter (_type_): data iterator
        seqs_iter (_type_): sequences iterator
        device (_type_): device of predicting

    Returns:
        _type_: 3-tuple: (prediction label, probability of positive, inferring motif)
    """
    model = model.to(device)
    model.eval()
    y_pred_all, y_pred_proba_all, pred_motifs_all = [], [], []
    with torch.no_grad():
        for dataset, seqs in zip(data_iter, seqs_iter):
            dataset = [dataset[i].to(device) for i in range(len(dataset))]
            outputs = model(dataset)
            # 预测标签
            y_pred = torch.max(outputs, 1)[1].cpu().numpy()
            y_pred_all.extend(y_pred)
            # 预测概率
            y_pred_proba = torch.softmax(outputs, 1).cpu().numpy()[:, 1]
            y_pred_proba_all.extend(y_pred_proba)
            # 获取多个头的注意力
            attention_weights = model.attention_wight2.cpu().detach().numpy()
            # 平均每个注意力头的权重
            seq_length = attention_weights.shape[-1]
            avg_attention_weights = attention_weights.reshape(
                len(dataset[0]), 4, seq_length, seq_length).mean(axis=1)
            # 计算每个位置的注意力分数和
            attention_scores = avg_attention_weights.sum(axis=1)
            # 根据重要位置进行基序推断
            pred_motifs = [format_motifs(get_important_regins(
                score[:len(seqs[i])])) for i, score in enumerate(attention_scores)]
            pred_motifs_all.extend(pred_motifs)
    return np.array(y_pred_all), np.array(y_pred_proba_all), np.array(pred_motifs_all)


def normalize(values):
    max_val, min_val = max(values), min(values)
    # 处理特殊情况：所有值都相等
    if min_val == max_val:
        return [1.0] * len(values)
    else:
        return [(val - min_val) / (max_val - min_val) for val in values]


def format_motifs(motifs):
    return '; '.join(['MOTIF {}..{}'.format(motif[0], motif[1]) for motif in motifs])


def get_important_regins(score):
    motif, regions = [], []
    threshold = 0.5
    k_extend = 2
    score = normalize(score)
    # 根据阈值确定每一个位置的重要程度
    importance = [1 if i > threshold else 0 for i in score]
    # 对于重要程度不为零的连续区域，记录其起始位置和终止位置
    start = -1
    for i in range(len(importance)):
        if importance[i] == 1:
            if start == -1:
                start = i
        else:
            if start != -1:
                regions.append((start, i-1))
                start = -1
    if start != -1:
        regions.append((start, len(importance)-1))
    # 扩展单个重要位置的区域（最后一个除外）
    regions_extend = [(t[0], t[1] + k_extend) for t in regions[:-1]]
    regions_extend.append(regions[-1])
    # 合并重叠的区域
    for i in range(len(regions_extend)):
        if len(motif) == 0 or motif[-1][1] < regions_extend[i][0]:
            motif.append(regions_extend[i])
        elif motif[-1][1] < regions_extend[i][1]:
            motif[-1] = (motif[-1][0], regions_extend[i][1])
    return motif
