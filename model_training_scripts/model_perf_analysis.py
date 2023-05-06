import numpy as np
import pandas as pd
import os

''' 
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