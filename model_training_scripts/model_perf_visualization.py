import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

# GLOBAL VARIABLES

pos_class = 'parasite'
unsure_ignored = True

data_dir = '../ann_with_predictions_r34_b32/s_3b'
trend_1 = 'FP'
trend_2 = 'FNR'
plot_title = 'Model 3B: ' + pos_class + ' prediction threshold vs. ' + trend_1 + ' and ' + trend_2

if unsure_ignored:
    model_performance_path = data_dir + '/model_r34_b32_performance_' + pos_class + '_v_rest_unsure_ignored.csv'
    plot_path = data_dir + '/model_r34_b32_performance_' + pos_class + '_v_rest_unsure_ignored_' + trend_1 + '_and_' + trend_2 + '.png'
else:
    model_performance_path = data_dir + '/model_r34_b32_performance_' + pos_class + '_v_rest.csv'
    plot_path = data_dir + '/model_r34_b32_performance_' + pos_class + '_v_rest_' + trend_1 + '_and_' + trend_2 + '.png'

model_perf_df = pd.read_csv(model_performance_path, index_col = 'index')
index_label = model_perf_df[model_perf_df['thresh'] == 0.8].index[0]
model_perf_df = model_perf_df.iloc[index_label:]
print(model_perf_df)

# VISUALIZATION

fig, ax1 = plt.subplots()
fig.suptitle(plot_title)

color = 'tab:red'
ax1.set_xlabel('Threshold')
ax1.set_ylabel(trend_1, color=color)
ax1.yaxis.set_major_locator(ticker.MaxNLocator(20))
ax1.plot(model_perf_df['thresh'], model_perf_df[trend_1], color=color, marker='o', markersize=1)
ax1.tick_params(axis='y', labelcolor=color)
plt.ticklabel_format(style='plain', axis='y')

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel(trend_2, color=color)
ax2.plot(model_perf_df['thresh'], model_perf_df[trend_2], color=color, marker='o', markersize=1)
ax2.tick_params(axis='y', labelcolor=color)
ax2.plot(marker='o', markersize=8)
plt.ticklabel_format(style='plain', axis='y')

fig.tight_layout()
plt.show()

fig.savefig(plot_path, dpi=300, bbox_inches='tight')




