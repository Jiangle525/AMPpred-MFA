import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import datetime


def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def colorful(s, font_color, mode_status):
    color_list = [
        'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'
    ]
    mode_status_dict = {
        'default': 0,
        'bold': 1,
        'underline': 4,
        'blink': 5,
        'inverse': 7
    }
    font_color_dict = {color_list[i]: 30 + i for i in range(len(color_list))}
    background_color_dict = {
        color_list[i]: 40 + i for i in range(len(color_list))}
    # return '\033[{};{};{}m{}\033[0m'.format(mode_status_dict[mode_status], font_color_dict[font_color], background_color_dict[background_color],s)
    return '\033[{};{}m{}\033[0m'.format(mode_status_dict[mode_status], font_color_dict[font_color], s)


def cover_print(s, newline):
    print('\r' + s, end='\n' if newline else '', flush=True)


def colorful_print(s, font_color='white', mode_status='default', is_cover_print=False, newline=True):
    if is_cover_print:
        cover_print(colorful(s, font_color, mode_status), newline)
    else:
        print(colorful(s, font_color, mode_status))


def draw_metrics(mtrcs):
    fig = plt.figure(figsize=(10, 5), facecolor='white')
    fig_trian = plt.subplot(1, 2, 1)
    fig_trian.plot(mtrcs['train acc'])
    fig_trian.plot(mtrcs['valid acc'])
    fig_trian.set_title('Train accuracy')
    fig_trian.set_ylabel('Accuracy')
    fig_trian.set_xlabel('Epoch')
    fig_trian.legend(['Train', 'Validation'], loc='upper left')

    fig_vali = plt.subplot(1, 2, 2)
    fig_vali.plot(mtrcs['train loss'])
    fig_vali.plot(mtrcs['valid loss'])
    fig_vali.set_title('Train loss')
    fig_vali.set_ylabel('Loss')
    fig_vali.set_xlabel('Epoch')
    fig_vali.legend(['Train', 'Validation'], loc='upper left')
    return fig


def draw_roc(fprs, tprs):
    """_summary_

    Args:
        fprs (_type_): list of fpr
        tprs (_type_): list of tpr

    Returns:
        _type_: fig
    """

    colors = [
        'darkorange', 'aqua', 'cornflowerblue', 'blueviolet', 'deeppink',
        'cyan'
    ]
    roc_fig = plt.figure()
    fig = plt.subplot(1, 1, 1)
    fig.set_title('ROC curve')
    fig.set_xlim([0.0, 1.0])
    fig.set_ylim([0.0, 1.0])
    fig.set_xlabel('False Positive Rate')
    fig.set_ylabel('True Positive Rate')
    fig.plot([0, 1], [0, 1],
             linestyle='--',
             lw=2,
             color='r',
             label='Random',
             alpha=0.8)

    aucs = []
    mean_fpr = np.linspace(0, 1, max([len(i) for i in fprs]))
    interp_tprs = []

    for i in range(len(fprs)):
        fig.plot(fprs[i],
                 tprs[i],
                 lw=1,
                 alpha=0.5,
                 color=colors[i % len(colors)],
                 label='{} fold (AUC = {:.3f})'.format(
                     i + 1, metrics.auc(fprs[i], tprs[i])))
        interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
        interp_tprs.append(interp_tpr)
        aucs.append(metrics.auc(fprs[i], tprs[i]))

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0, 1
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    fig.plot(mean_fpr,
             mean_tpr,
             color='b',
             label=r'Mean (AUC = {:.3f} ± {:.3f})'.format(mean_auc, std_auc),
             lw=2,
             alpha=0.8)

    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    fig.fill_between(mean_fpr,
                     tprs_lower,
                     tprs_upper,
                     color='slategray',
                     alpha=0.2,
                     label=r'± 1 std. dev.')

    fig.legend(loc="lower right")

    return roc_fig
