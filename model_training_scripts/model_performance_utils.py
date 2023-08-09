import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def model_perf_analysis(ann_w_pred_path, out_model_performance_path, pos_class='parasite', unsure_ignored=True):

    ann_w_pred_df = pd.read_csv(ann_w_pred_path, index_col = 'index')

    if unsure_ignored:
        # if I want to ignore all unsure for performance analysis:
        ann_dict = {'non-parasite':0, 'parasite':1}
    else:
        # if I want to include the unsure in the counts, but assume they're negative for performance analysis:
        ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2}

    # BASIC COUNTS
    thresh_delta = 0.002
    thresh_vals = np.arange(0, 1 + thresh_delta, thresh_delta) # from 0 to 1 in 0.01 increments

    # P, N counts
    class_nums = np.zeros(len(ann_dict))
    pos_num = 0
    neg_num = 0
    for i, key in enumerate(ann_dict):
        class_nums[i] = (ann_w_pred_df['annotation'] == ann_dict[key]).sum()
        if key == pos_class:
            pos_num = class_nums[i]
        else:
            neg_num += class_nums[i]

    # TN TP FN FP counts
    perf_columns = ['thresh', 'TP', 'TN', 'FP', 'FN', 'predicted pos', 'predicted neg'] # add more cols
    perf_df = pd.DataFrame(columns=perf_columns)

    cond_pos = ann_w_pred_df['annotation'] == ann_dict[pos_class]
    cond_neg = np.any([(ann_w_pred_df['annotation'] == val) for val in ann_dict.values() if val != ann_dict[pos_class]],axis=0)
    for i in range(len(thresh_vals)):
        cond_pred_pos = ann_w_pred_df[pos_class + ' output'] > thresh_vals[i]
        
        tp = float(len(ann_w_pred_df[cond_pos & cond_pred_pos]))
        tn = float(len(ann_w_pred_df[cond_neg & ~cond_pred_pos]))
        fp = float(len(ann_w_pred_df[cond_neg & cond_pred_pos]))
        fn = float(len(ann_w_pred_df[cond_pos & ~cond_pred_pos]))
        pred_pos = tp + fp
        pred_neg = tn + fn

        perf_df.loc[i, perf_columns] = [thresh_vals[i],tp,tn,fp,fn,pred_pos,pred_neg]

    c = len(ann_w_pred_df['annotation']) # total number of images
    print(class_nums)

    # performance calculations

    # accuracy (TP + TN / all)
    perf_df['accuracy'] = (perf_df['TP'] + perf_df['TN']) / c

    # recall, sensitivity, or true positive rate, TPR (# TP / # pos)
    perf_df['TPR'] = (perf_df['TP']) / pos_num

    # specificity, or true negative rate (# TN / # neg)
    perf_df['TNR'] = perf_df['TN'] / neg_num

    # false positive rate, FPR (# FP / # neg)
    perf_df['FPR'] = perf_df['FP'] / neg_num

    # false negative rate, FNR, or miss rate (# FN / # pos)
    perf_df['FNR'] = perf_df['FN'] / pos_num

    # precision, or PPV (# TP / # pred pos)
    ppv_mask = perf_df['predicted pos'] != 0
    perf_df.loc[ppv_mask,'PPV'] = perf_df.loc[ppv_mask,'TP'] / perf_df.loc[ppv_mask,'predicted pos']
    perf_df.loc[~ppv_mask,'PPV'] = np.inf

    # NPV (# TN / # pred neg)
    npv_mask = perf_df['predicted neg'] != 0
    perf_df.loc[npv_mask,'NPV'] = perf_df.loc[npv_mask,'TN'] / perf_df.loc[npv_mask,'predicted neg']
    perf_df.loc[~npv_mask,'NPV'] = np.inf

    # false discovery rate, FDR (# FP / # pred pos)
    fdr_mask = perf_df['predicted pos'] != 0
    perf_df.loc[fdr_mask,'FDR'] = perf_df.loc[fdr_mask,'FP'] / perf_df.loc[fdr_mask,'predicted pos']
    perf_df.loc[~fdr_mask,'FDR'] = np.inf

    # false omission rate, FOR (# FN / # pred neg)
    for_mask = perf_df['predicted neg'] != 0
    perf_df.loc[for_mask,'FOR'] = perf_df.loc[for_mask,'FN'] / perf_df.loc[for_mask,'predicted neg']
    perf_df.loc[~for_mask,'FOR'] = np.inf

    # F1 score, when both false pos and neg are bad (2 * # TP / 2 * # TP + # FP + # FN)
    f1_mask = 2*perf_df['TP'] + perf_df['FP'] + perf_df['FN'] != 0
    perf_df.loc[f1_mask,'F1'] = 2 * perf_df.loc[f1_mask,'TP'] / (2 * perf_df.loc[f1_mask,'TP'] + perf_df.loc[f1_mask,'FP'] + perf_df.loc[f1_mask,'FN'])
    perf_df.loc[~f1_mask,'F1'] = np.inf

    # positive likelihood ratio (TPR / FPR)
    lrp_mask = perf_df['FPR'] != 0
    perf_df.loc[lrp_mask,'LR+'] = perf_df.loc[lrp_mask,'TPR'] / perf_df.loc[lrp_mask,'FPR']
    perf_df.loc[~lrp_mask,'LR+'] = np.inf

    # negative likelihood ratio (FNR / TNR)
    lrn_mask = perf_df['TNR'] != 0
    perf_df.loc[lrn_mask,'LR-'] = perf_df.loc[lrn_mask,'FNR'] / perf_df.loc[lrn_mask,'TNR']
    perf_df.loc[~lrn_mask,'LR-'] = np.inf

    # Matthews correlation coefficient, MCC (see wiki link)
    # (TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # https://en.wikipedia.org/wiki/Phi_coefficient
    # TODO: look into what this is
    mcc_mask = (perf_df['predicted neg'] != 0) & (perf_df['predicted pos'] != 0)
    perf_df.loc[mcc_mask,'MCC'] = perf_df.loc[mcc_mask,'TP']*perf_df.loc[mcc_mask,'TN'] - perf_df.loc[mcc_mask,'FP']*perf_df.loc[mcc_mask,'FN']
    perf_df.loc[mcc_mask,'MCC'] = perf_df.loc[mcc_mask,'MCC'] / (perf_df.loc[mcc_mask,'predicted pos']*perf_df.loc[mcc_mask,'predicted neg']*(perf_df.loc[mcc_mask,'TP']+perf_df.loc[mcc_mask,'FN'])*(perf_df.loc[mcc_mask,'TN']+perf_df.loc[mcc_mask,'FP'])).fillna(1).apply(np.sqrt)
    perf_df.loc[~mcc_mask,'MCC'] = np.inf

    # Fowlkes-Mallows index, FM (see wiki link)
    # sqrt(PPV*TPR)
    # https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
    # TODO: look into what this is
    fm_mask = perf_df['predicted pos'] != 0
    perf_df.loc[fm_mask,'FM'] = (perf_df.loc[fm_mask,'PPV']*perf_df.loc[fm_mask,'TPR']).fillna(1).apply(np.sqrt)
    perf_df.loc[~fm_mask,'FM'] = np.inf

    # Youden's J statistic (see wiki link)
    # TPR + TNR - 1
    # https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    # TODO: look into what this is
    perf_df['J'] = perf_df['TPR'] + perf_df['TNR'] - 1

    # Diagnostic odds ratio, DOR (LR+/LR-)
    # https://en.wikipedia.org/wiki/Diagnostic_odds_ratio
    # TODO: look into what this is
    dor_mask = ((perf_df['FPR'] != 0) & (perf_df['TNR'] != 0)) & (perf_df['LR-'] != 0)
    perf_df.loc[dor_mask,'DOR'] = perf_df.loc[dor_mask,'LR+'] / perf_df.loc[dor_mask,'LR-']
    perf_df.loc[~dor_mask,'DOR'] = np.inf

    perf_df.replace(np.inf, None)
    perf_df.index.name = 'index'
    perf_df.to_csv(out_model_performance_path)

def model_perf_visualization(dataset_id, model_performance_path, out_trend1_v_trend2_path, trend_1='FP', trend_2='FNR', pos_class='parasite'):

    # trends to plot: FP and FNR
    trend_1 = 'FP'
    trend_2 = 'FNR'

    # title of plot
    plot_title = dataset_id + ': ' + pos_class + ' prediction threshold vs. ' + trend_1 + ' and ' + trend_2

    perf_df = pd.read_csv(model_performance_path, index_col = 'index')

    # visualization

    fig, ax1 = plt.subplots()
    fig.suptitle(plot_title)

    color = 'tab:red'
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel(trend_1, color=color)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(20))
    ax1.plot(perf_df['thresh'], perf_df[trend_1], color=color, marker='o', markersize=1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel(trend_2, color=color)
    ax2.plot(perf_df['thresh'], perf_df[trend_2], color=color, marker='o', markersize=1)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(marker='o', markersize=8)
    plt.ticklabel_format(style='plain', axis='y')

    fig.tight_layout()
    plt.show()

    fig.savefig(out_trend1_v_trend2_path, dpi=300, bbox_inches='tight')