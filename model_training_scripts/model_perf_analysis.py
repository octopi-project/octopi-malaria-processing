import numpy as np
import pandas as pd
import os

# GLOBAL VARIABLES

data_dir = data_dir = '/media/rinni/Extreme SSD/Rinni/to-combine/s_a/'
ann_w_pred_path = '/ann_with_predictions.csv'
relabeled_ann = True # make this the ifdef thing

ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2, 'unlabeled':-1}

thresh_delta = 0.05
thresh_vals = np.arange(0, 1 + thresh_delta, thresh_delta) # from 0 to 1 in 0.05 increments

# P, N counts
ann_w_pred_df = pd.read_csv(data_dir + ann_w_pred_path, index_col = 'index')
neg_num = (ann_w_pred_df['annotation'] == 0).sum()
pos_num = (ann_w_pred_df['annotation'] == 1).sum()

# TN TP FN FP counts
perf_columns = ['thresh', 'TP', 'TN', 'FP', 'FN'] # add more cols
perf_df = pd.DataFrame(columns=perf_columns)

for i in range(len(thresh_vals)):
    cond_neg = ann_w_pred_df['annotation'] == 0
    cond_pos = ann_w_pred_df['annotation'] == 1
    cond_pred_pos = ann_w_pred_df['parasite output'] > thresh_vals[i]
    
    tp = float(len(ann_w_pred_df[cond_pos & cond_pred_pos]))
    tn = float(len(ann_w_pred_df[cond_neg & ~cond_pred_pos]))
    fp = float(len(ann_w_pred_df[cond_neg & cond_pred_pos]))
    fn = float(len(ann_w_pred_df[cond_pos & ~cond_pred_pos]))

    perf_df.loc[i, perf_columns] = [thresh_vals[i],tp,tn,fp,fn]
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
perf_df['FPR'] = 1 - perf_df['TNR']

# false negative rate, FNR, or miss rate (# FN / # pos)
perf_df['FNR'] = perf_df['FN'] / pos_num

# precision, or PPV (# TP / # pred pos)
perf_df['PPV'] = np.divide(perf_df['TP'], perf_df['TP'] + perf_df['FP'], where = perf_df['TP'] + perf_df['FP'] != 0)

# NPV (# TN / # pred neg)
perf_df['NPV'] = np.divide((perf_df['TN']), (perf_df['TN'] + perf_df['FN']), where = (perf_df['TN'] + perf_df['FN']) != 0)

# false discovery rate, FDR (# FP / # pred pos)
perf_df['FDR'] = np.subtract(1, perf_df['PPV'], where = pd.notnull(perf_df['PPV']))

# false omission rate, FOR (# FN / # 
perf_df['FOR'] = np.subtract(1, perf_df['NPV'], where = pd.notnull(perf_df['NPV']))

# F1 score (when both false pos and neg are bad)
perf_df['F1'] = np.divide((2.0 * perf_df['PPV'] * perf_df['TPR']), (perf_df['PPV'] + perf_df['TPR']), where = (perf_df['TP'] != 0) & (perf_df['PPV'] is not None))
print(perf_df)

# positive likelihood ratio (TPR / FPR)
perf_df['LR+'] = np.divide(perf_df['TPR'], perf_df['FPR'], where = perf_df['FPR'] != 0)

# negative likelihood ratio (FNR / TNR)
perf_df['LR-'] = np.divide(perf_df['FNR'], perf_df['TNR'], where = perf_df['TNR'] != 0)

# Matthews correlation coefficient, MCC (see wiki link)
# (TP*TN-FP*FN)/((TP+FP)(TP+FN)(TN+FP)(TN+FN))
# https://en.wikipedia.org/wiki/Phi_coefficient
# TODO: look into what this is
perf_df['MCC'] = np.divide(perf_df['TP']*perf_df['TN'] - perf_df['FP']*perf_df['FN'],(np.sqrt((perf_df['TP']+perf_df['FP'])*(perf_df['TP']+perf_df['FN'])*(perf_df['TN']+perf_df['FP'])*(perf_df['TN']+perf_df['FN'])), where = (perf_df['TP']+perf_df['FP'])*(perf_df['TP']+perf_df['FN'])*(perf_df['TN']+perf_df['FP'])*(perf_df['TN']+perf_df['FN'])
 != 0)

# Fowlkes-Mallows index, FM (see wiki link)
# sqrt(PPV*TPR)
# https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
# TODO: look into what this is
perf_df['FM'] = np.sqrt(perf_df['PPV']*perf_df['TPR'])

# Youden's J statistic (see wiki link)
# TPR + TNR - 1
# https://en.wikipedia.org/wiki/Youden%27s_J_statistic
# TODO: look into what this is
perf_df['J'] = perf_df['TPR'] + perf_df['TNR'] - 1

# Diagnostic odds ratio, DOR (LR+/LR-)
# https://en.wikipedia.org/wiki/Diagnostic_odds_ratio
# TODO: look into what this islr
perf_df['DOR'] = perf_df['LR+']/perf_df['LR-']

print(perf_df)

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