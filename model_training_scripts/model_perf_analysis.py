import numpy as np
import pandas as pd
import os

''' 
want to take in ann with predictions and:
if 2 class classifier, I want a few things:
- a = frac of unsure labeled pos
- b = frac of pos labeled pos
- c = frac of neg labeled pos
number of pos, neg, uns: d, e, f
also, if I want to run the analysis on the relabeled annotations, I can sub out the annotations column in my file with the annotations column in the relabeled file (should be in the same order!)
then, I want:
- b_1 = frac of pos labeled pos in relabeled
- c_1 = frac of neg labeled pos in relabeled
number of pos, neg in relabeled: d_1, e_1
as I vary the positive threshold (the value t at which pred > t means we predict positive)
note that:
- FPR = (a + c)/(e) or c_1/(e_1)
- TPR = b/d or b_1/d_1
plot ROC: FPR (x) vs TPR (y)

for a given threshold, calculate:
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
