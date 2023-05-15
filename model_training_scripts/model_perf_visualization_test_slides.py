import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os

# GLOBAL VARIABLES

pos_class = 'parasite'
unsure_ignored = True

classifier = 's_a'
model_arch = '_r34_b32'
fpr_thresh = 0.876
data_dir = '/Users/rinnibhansali/Documents/Stanford/Research/Thesis/perf-to-export/' + classifier + '/neg_slides_predictions/'
trend_1 = 'FPR'
trend_2 = 'FNR'
plot_title = 'Resnet34 A: ' + pos_class + ' prediction threshold vs. ' + trend_1 + ' and ' + trend_2

fpr_dict = {}
for model_performance_path in os.listdir(data_dir):
    # Check if file name contains the model architecture
    if (model_arch in model_performance_path and 'performance' in model_performance_path) and 'png' not in model_performance_path:
        if unsure_ignored:
            plot_path = data_dir + model_performance_path.split("_model")[0] + '_model' + model_arch + '_' + pos_class + '_v_rest_unsure_ignored_' + trend_1 + '_and_' + trend_2 + '.png'
            print(plot_path)
        else:
            plot_path = data_dir + model_performance_path.split("_model")[0] + '_model' + model_arch + '_' + pos_class + '_v_rest_' + trend_1 + '_and_' + trend_2 + '.png'

        model_perf_df = pd.read_csv(data_dir + model_performance_path, index_col = 'index')
        index_label = model_perf_df[model_perf_df['thresh'] == 0.8].index[0]
        model_perf_df = model_perf_df.iloc[index_label:]
        print(model_perf_df)

        fpr_dict[model_performance_path.split("_model")[0]] = model_perf_df.loc[model_perf_df['thresh'] == fpr_thresh, 'FPR'].values[0]

        # VISUALIZATION

        fig, ax1 = plt.subplots()
        fig.suptitle(plot_title)

        color = 'tab:red'
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel(trend_1, color=color)
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(20))
        ax1.plot(model_perf_df['thresh'], model_perf_df[trend_1], color=color, marker='o', markersize=1)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel(trend_2, color=color)
        ax2.plot(model_perf_df['thresh'], model_perf_df[trend_2], color=color, marker='o', markersize=1)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(marker='o', markersize=8)
        plt.ticklabel_format(style='plain', axis='y')

        fig.tight_layout()

        fig.savefig(plot_path, dpi=300, bbox_inches='tight')

# extract slide names and FPR values
print(fpr_dict.keys())
fig, ax = plt.subplots()

ax.bar(fpr_dict.keys(), fpr_dict.values())
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xticklabels(range(len(fpr_dict)))

plt.title('FPR on all negative test slides')
plt.xlabel('Slides')
plt.ylabel('FPR')
plt.xticks(fontsize=8)

fig.tight_layout()

fig.savefig(data_dir + '/fpr_on_all_slides_' + classifier + '_' + model_arch + '.png', dpi=300, bbox_inches='tight')
print(fpr_dict.values())