import numpy as np
import pandas as pd
import os

# GLOBAL VARIABLES

data_dir = data_dir = '/media/rinni/Extreme SSD/Rinni/to-combine/s_3b/'
ann_w_pred_path = '/ann_with_predictions_cl.csv'
relabeled_ann = True # make this the ifdef thing

ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2, 'unlabeled':-1}

thresh_delta = 0.05
thresh_vals = np.arange(0, 1 + thresh_delta, thresh_delta) # from 0 to 1 in 0.05 increments

# P, N counts
ann_w_pred_df = pd.read_csv(data_dir + ann_w_pred_path, index_col = 'index')
neg_num = (ann_w_pred_df['annotation'] == 0).sum()
pos_num = (ann_w_pred_df['annotation'] == 1).sum()

# TN TP FN FP counts
perf_columns = ['thresh', 'TP', 'TN', 'FP', 'FN', 'predicted pos', 'predicted neg'] # add more cols
perf_df = pd.DataFrame(columns=perf_columns)

for i in range(len(thresh_vals)):
    cond_neg = ann_w_pred_df['annotation'] == 0
    cond_pos = ann_w_pred_df['annotation'] == 1
    cond_pred_pos = ann_w_pred_df['parasite output'] > thresh_vals[i]
    
    tp = float(len(ann_w_pred_df[cond_pos & cond_pred_pos]))
    tn = float(len(ann_w_pred_df[cond_neg & ~cond_pred_pos]))
    fp = float(len(ann_w_pred_df[cond_neg & cond_pred_pos]))
    fn = float(len(ann_w_pred_df[cond_pos & ~cond_pred_pos]))
    pred_pos = tp + fp
    pred_neg = tn + fn

    perf_df.loc[i, perf_columns] = [thresh_vals[i],tp,tn,fp,fn,pred_pos,pred_neg]
    print('For threshold ' + str(round(thresh_vals[i],2)) + ': TP = ' + str(round(tp,2)) + ', TN = ' + str(round(tn,2)) + ', FP = ' + str(round(fp,2)) + ', FN = ' + str(round(fn,2)))

c = len(ann_w_pred_df['annotation']) # total number of images

# Performance calculations

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
perf_df.to_csv(data_dir + '/model_performance_maybe_s_3b.csv')
'''
if relabeled_ann:
    # do it all again
    ann_relabeled_path = '/unsure_relabeled_thresh_0.9.csv'

    ann_relabeled_df = pd.read_csv(ann_relabeled_path, index = 'index')
    ann_w_pred_relabeled_df = ann_w_pred_df.copy()
    ann_w_pred_relabeled_df['annotation'] = ann_relabeled_df['annotation']
'''

""" 
Binary classifier:
for a given threshold, (the value t at which para > t means we predict positive) calculate:
- a = frac of unsure labeled pos
- b = frac of pos labeled pos
- c = frac of neg labeled pos
- number of pos, neg, uns: d, e, f
- b_1 = frac of pos labeled pos in relabeled
- c_1 = frac of neg labeled pos in relabeled
- number of pos, neg in relabeled: d_1, e_1
- FPR
- TPR
- FNR
- TNR
- FP
- FN
- TP
- TN
- Sensitivity
- Specificity

Ternary classifier:
One way:
Pick a threshold again (the value t at which para pred > t means we predict positive)
For all other images, predict negative
Then calculate the same things!
Other way:
check that link, but can do the one vs rest strategy above for all classes^!

pseudocode:
make df where rows are for each new thresh t and cols are values of interest
define it with column names^ (thres, TN, TP, ..., specificity or something)
begin for loop, running through thresh t
read in csv
get np array of predictions based on thresh t
compare predictions to ground truth
get TN, TP, FN, FP
calculate all other values accordingly
fill into df as you go 
"""