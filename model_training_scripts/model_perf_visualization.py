import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

# GLOBAL VARIABLES

pos_class = 'parasite'
multiclass_and_unsure_ignored = True

classifier = 's_a'
model_arch = '_r18_b32'
data_dir ='../perf-to-export/' + classifier + '/test_predictions/'
trend_1 = 'FPR'
trend_2 = 'FNR'
plot_title = 'Resnet18 A: ' + pos_class + ' prediction threshold vs. ' + trend_1 + ' and ' + trend_2

if multiclass_and_unsure_ignored:
    model_performance_path = data_dir + '/model' + model_arch + '_performance_' + pos_class + '_v_rest_unsure_ignored.csv'
    plot_path = data_dir + '/model' + model_arch + '_' + pos_class + '_v_rest_unsure_ignored_' + trend_1 + '_and_' + trend_2 + '.png'
else:
    model_performance_path = data_dir + '/model' + model_arch + '_performance_' + pos_class + '_v_rest.csv'
    plot_path = data_dir + '/model' + model_arch + '_' + pos_class + '_v_rest_' + trend_1 + '_and_' + trend_2 + '.png'

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
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

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




