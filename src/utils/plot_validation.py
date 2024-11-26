import json
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from utils.plotting import ENV_SHOW_PLOT

def flatten(list_of_lists):
    out = []
    for i in range(len(list_of_lists)):
        for j in range(len(list_of_lists[i])):
            out.append(list_of_lists[i][j])

    return out


def plot_validation(parent_dir: Path,
                    metrics_file_name: Path | str,
                    plot_save_dir: Path):

    model_names: list[str] = []  # list of model names
    losses: list[list[float]] = []  # list of list of loss from each fold
    top1_accs: list[list[float]] = []  # list of list of top 1 accuracy from each fold
    top3_accs: list[list[float]] = []  # list of list of top 3 accuracy from each fold
    aurocs: list[list[float]] = []  # list of list of aurocs

    avg_losses: list[float] = []
    avg_acc1: list[float] = []
    avg_acc3: list[float] = []
    avg_acc5: list[float] = []
    avg_recall: list[float] = []
    avg_precision: list[float] = []
    avg_auroc: list[float] = []
    avg_f_one: list[float] = []

    micro_acc1: list[float] = []
    micro_acc3: list[float] = []
    micro_acc5: list[float] = []
    micro_recall: list[float] = []
    micro_precision: list[float] = []
    micro_f_one: list[float] = []


    # scour the directory for metrics file name
    sub_directories = [f for f in parent_dir.iterdir() if f.is_dir()]
    for sub_dir in sub_directories:
        if not (sub_dir / metrics_file_name).exists():
            # if does not contain metric then skip
            continue

        with sub_dir.joinpath(metrics_file_name).open() as metrics_file:
            metrics_data = metrics_file.read()
            metrics_dict = json.loads(metrics_data)

        model_names.append(metrics_dict['Model Name'])
        losses.append(metrics_dict['folds_losses'])
        top1_accs.append(metrics_dict['folds_micro_acc1'])
        top3_accs.append(metrics_dict['folds_micro_acc3'])
        aurocs.append(metrics_dict['folds_auroc'])

        avg_losses.append(metrics_dict['avg_loss'])
        avg_acc1.append(metrics_dict['avg_acc1'])
        avg_acc3.append(metrics_dict['avg_acc3'])
        avg_acc5.append(metrics_dict['avg_acc5'])
        avg_recall.append(metrics_dict['avg_recall'])
        avg_precision.append(metrics_dict['avg_precision'])
        avg_auroc.append(metrics_dict['avg_auroc'])
        avg_f_one.append(metrics_dict['avg_f_one'])

        micro_acc1.append(metrics_dict['avg_micro_acc1'])
        micro_acc3.append(metrics_dict['avg_micro_acc3'])
        micro_acc5.append(metrics_dict['avg_micro_acc5'])

    # build pandas dataframe
    metrics_df = pd.DataFrame({'Model Name': model_names,
                               'Loss': avg_losses,
                               'Top-1 Acc': avg_acc1,
                               'Top-3 Acc': avg_acc3,
                               'Top-5 Acc': avg_acc5,
                               'Recall': avg_recall,
                               'Precision': avg_precision,
                               'AUROC': avg_auroc,
                               'F1 Score': avg_f_one,

                               'Micro Acc1': micro_acc1,
                               'Micro Acc3': micro_acc3,
                               'Micro Acc5': micro_acc5})

    # find data shapes
    models_num = len(model_names)
    folds_num = len(losses[0])  # assume everything has the same fold

    # model name column
    model_names_col = []
    for i in range(models_num):
        model_names_col += ([model_names[i]] * len(losses[i]))

    # inset plot params
    x_start = -.25
    x_width = models_num - 0.5

    # Facet plot
    sn.set_color_codes("deep")
    sn.set_theme(style="whitegrid")
    colors = sn.color_palette("Paired")
    font = {'size': 16}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(30, 20))
    fig.suptitle('K-Fold Validation', fontsize=50)

    # acc1 plot
    ax1 = fig.add_subplot(231)
    ax1.set_title("Top-1 Accuracy", fontsize=40)
    ax1.set_xticklabels(model_names, rotation=45, ha="right", fontsize=16)

    # build dictionary
    acc_dict = {
        'Model Name': model_names_col,
        'Accuracy (%)': [flatten(top1_accs)[i] * 100 for i in range(len(flatten(top1_accs)))],
    }
    acc_df = pd.DataFrame(acc_dict)

    # violin plot codes
    # sn.violinplot(acc_df, x="Model Name", y="Accuracy (%)", color="g", ax=ax2, inner=None, fill=False, width=1, linewidth=5)
    # sn.swarmplot(acc_df, x="Model Name", y="Accuracy (%)", color="g", size=10, ax=ax2)

    sn.barplot(data=acc_df, x="Model Name", y="Accuracy (%)", errorbar="sd", color=colors[0], ax=ax1, err_kws={'color': colors[1]})
    plt.ylim(ymax=100)
    ax1.set(xlabel=None)
    ax1.tick_params(axis='y', which='major', labelsize=14)
    ax1.tick_params(axis='x', bottom=True)
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=24)

    # zoomed acc1 plot
    # loc values
    # 2---8---1
    # |   |   |
    # 6---10--5/7
    # |   |   |
    # 3---9---4
    axins1 = inset_axes(ax1, width="50%", height="40%", loc="lower right")
    sn.barplot(data=acc_df, x="Model Name", y="Accuracy (%)", errorbar="sd", color=colors[0], ax=axins1, err_kws={'color': colors[1]})
    acc1_min = np.min(avg_acc1)
    acc1_max = np.max(avg_acc1)
    acc1_ymin = (acc1_min * 100) - 5
    acc1_ymax = (acc1_max * 100) + 5
    axins1.set_ylim(acc1_ymin, acc1_ymax)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    axins1.set(xlabel=None, ylabel=None)
    axins1.bar_label(axins1.containers[0], size=10, fmt="{:.02f}%", label_type='center', weight='bold')
    ins1 = ax1.indicate_inset((x_start, acc1_ymin, x_width, acc1_ymax - acc1_ymin), axins1)
    ins1.connectors[0].set_visible(True)
    ins1.connectors[1].set_visible(False)
    ins1.connectors[2].set_visible(False)
    ins1.connectors[3].set_visible(True)

    # acc3 plot
    ax2 = fig.add_subplot(232)
    ax2.set_title("Top-3 Accuracy", fontsize=40)
    ax2.set_xticklabels(model_names, rotation=45, ha="right", fontsize=16)
    # build dictionary
    acc3_dict = {
        'Model Name': model_names_col,
        'Accuracy (%)': [flatten(top3_accs)[i] * 100 for i in range(len(flatten(top3_accs)))],
    }
    acc3_df = pd.DataFrame(acc3_dict)
    sn.barplot(data=acc3_df, x="Model Name", y="Accuracy (%)", errorbar="sd", color=colors[2], ax=ax2, err_kws={'color': colors[3]})
    plt.ylim(ymax=100)
    ax2.set(xlabel=None)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax2.tick_params(axis='x', bottom=True)
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=24)

    axins2 = inset_axes(ax2, width="50%", height="40%", loc="lower right")
    sn.barplot(data=acc3_df, x="Model Name", y="Accuracy (%)", errorbar="sd", color=colors[2], ax=axins2, err_kws={'color': colors[3]})
    acc3_min = np.min(avg_acc3)
    acc3_max = np.max(avg_acc3)
    acc3_ymin = (acc3_min * 100) - 5
    acc3_ymax = (acc3_max * 100) + 5
    axins2.set_ylim(acc3_ymin, acc3_ymax)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    axins2.set(xlabel=None, ylabel=None)
    axins2.bar_label(axins2.containers[0], size=10, fmt="{:.02f}%", label_type='center', weight='bold')
    ins2 = ax2.indicate_inset((x_start, acc3_ymin, x_width, acc3_ymax - acc3_ymin), axins2)
    ins2.connectors[0].set_visible(True)
    ins2.connectors[1].set_visible(False)
    ins2.connectors[2].set_visible(False)
    ins2.connectors[3].set_visible(True)

    # auroc plot
    ax3 = fig.add_subplot(233)
    ax3.set_title("Area Under ROC", fontsize=40)
    ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=16)
    # build dict
    auroc_dict = {
        'Model Name': model_names_col,
        'AUROC' : flatten(aurocs)
    }
    auroc_df = pd.DataFrame(auroc_dict)
    sn.barplot(data=auroc_df, x="Model Name", y="AUROC", color=colors[4], ax=ax3, errorbar="sd", err_kws={'color': colors[5]})
    plt.ylim(ymax=1.0)
    ax3.set(xlabel=None)
    ax3.tick_params(axis='y', which='major', labelsize=14)
    ax3.tick_params(axis='x', bottom=True)
    ax3.set_ylabel(ax3.get_ylabel(), fontsize=24)

    # inset plot
    axins3 = inset_axes(ax3, width="50%", height="40%", loc="lower right")
    sn.barplot(data=auroc_df, x="Model Name", y="AUROC", errorbar="sd", color=colors[4], ax=axins3, err_kws={'color': colors[5]})
    auroc_ymin = np.min(avg_auroc) - 0.025
    auroc_ymax = np.max(avg_auroc) + 0.025
    axins3.set_ylim(auroc_ymin, auroc_ymax)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    axins3.set(xlabel=None, ylabel=None)
    axins3.bar_label(axins3.containers[0], size=10, fmt="{:.4f}", label_type='center', weight='bold')
    ins3 = ax3.indicate_inset((x_start, auroc_ymin, x_width, auroc_ymax - auroc_ymin), axins3)
    ins3.connectors[0].set_visible(True)
    ins3.connectors[1].set_visible(False)
    ins3.connectors[2].set_visible(False)
    ins3.connectors[3].set_visible(True)

    # table
    ax4 = fig.add_subplot(212)
    box = ax4.get_position()
    box.y0 = box.y0 - 0.15
    box.y1 = box.y1 - 0.15
    ax4.set_position(box)
    ax4.set_title("Model Validation Metrics", fontsize=40)
    ax4.set_axis_off()

    table_df = pd.DataFrame()
    table_df['Model Name'] = model_names
    table_df['Loss'] = metrics_df['Loss'].map('{:.5f}'.format)
    table_df['Top-1 Acc'] = metrics_df['Micro Acc1'].map('{:.3f}'.format) + metrics_df['Top-1 Acc'].map('({:.3f})'.format)
    table_df['Top-3 Acc'] = metrics_df['Micro Acc3'].map('{:.3f}'.format) + metrics_df['Top-3 Acc'].map('({:.3f})'.format)
    table_df['Top-5 Acc'] = metrics_df['Micro Acc5'].map('{:.3f}'.format) + metrics_df['Top-5 Acc'].map('({:.3f})'.format)
    table_df['Recall'] =  metrics_df['Recall'].map('{:.3f}'.format)
    table_df['Precision'] = metrics_df['Precision'].map('{:.3f}'.format)
    table_df['AUROC'] = metrics_df['AUROC'].map('{:.3f}'.format)
    table_df['F1 Score'] = metrics_df['F1 Score'].map('{:.3f}'.format)

    # metrics_df.loc[:, 'Loss'] = metrics_df['Loss'].map('{:.5f}'.format)
    # metrics_df.loc[:, 'Top-1 Acc'] = metrics_df['Top-1 Acc'].map('{:.3f}'.format)
    # metrics_df.loc[:, 'Top-3 Acc'] = metrics_df['Top-3 Acc'].map('{:.3f}'.format)
    # metrics_df.loc[:, 'Top-5 Acc'] = metrics_df['Top-5 Acc'].map('{:.3f}'.format)
    # metrics_df.loc[:, 'Recall'] =  metrics_df['Recall'].map('{:.3f}'.format)
    # metrics_df.loc[:, 'Precision'] = metrics_df['Precision'].map('{:.3f}'.format)
    # metrics_df.loc[:, 'AUROC'] = metrics_df['AUROC'].map('{:.3f}'.format)
    # metrics_df.loc[:, 'F1 Score'] = metrics_df['F1 Score'].map('{:.3f}'.format)

    table = pd.plotting.table(ax4, table_df, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(26)
    table.scale(1.5, 4)
    table.auto_set_column_width(col=list(range(len(table_df.columns))))

    # save plot
    plt.savefig(plot_save_dir)

    if ENV_SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    plot_validation(Path(__file__).parent.parent / "validation_models",
                    "k_fold_validation.json",
                    Path(__file__).parent.parent / "validation_models" / "validation_figs",)








