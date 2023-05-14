'''
gpu: 
cropping took 4.5299530029296875e-06s
removing background took 0.2896451950073242s
detecting spots took 0.0637359619140625s
processing spots took 1.6242549419403076s
boxing spots took 0.13513588905334473s
everything took 2.11285138130188s

not gpu:
cropping took 3.814697265625e-06s
removing background took 7.937226057052612s
detecting spots took 0.1901416778564453s
processing spots took 0.7626736164093018s
boxing spots took 0.10422420501708984s
everything took 8.994367837905884s
'''
import matplotlib.pyplot as plt
import numpy as np
import math

gpu_dict = {}
gpu_dict['cropping'] = 3.814697265625/1000000
gpu_dict['background removal'] = 0.2896451950073242
gpu_dict['spot detection'] = 0.0637359619140625
gpu_dict['spot processing'] = 1.6242549419403076
gpu_dict['total time'] = 2.11285138130188 - 0.13513588905334473

no_gpu_dict = {}
no_gpu_dict['cropping'] = 4.5299530029296875/1000000
no_gpu_dict['background removal'] = 7.937226057052612
no_gpu_dict['spot detection'] = 0.1901416778564453
no_gpu_dict['spot processing'] = 0.7626736164093018
no_gpu_dict['total time'] = 8.994367837905884 - 0.10422420501708984

print(str(no_gpu_dict['total time']/gpu_dict['total time']))

# Create x-axis labels from the keys in both dictionaries
x_labels = list(gpu_dict.keys())

# Create arrays of values for each dictionary, in the order of the x-axis labels
y_values_dict1 = [gpu_dict[label] for label in x_labels]
y_values_dict2 = [no_gpu_dict[label] for label in x_labels]

# y_values_dict1 = [math.log(value) for value in y_values_dict1]
# y_values_dict2 = [math.log(value) for value in y_values_dict2]

# Set the width of each bar
bar_width = 0.35

# Set the positions of the bars on the x-axis
x_positions = np.arange(len(x_labels))

# Create the plot
fig, ax = plt.subplots()
rects1 = ax.bar(x_positions - bar_width/2, y_values_dict1, bar_width, color='r', label='GPU')
rects2 = ax.bar(x_positions + bar_width/2, y_values_dict2, bar_width, color='b', label='No GPU')

# Add some text for labels, title and axes ticks
ax.set_xlabel('Keys')
ax.set_ylabel('Time')
ax.set_title('Comparison of Processing Time: GPU vs. No GPU Support')
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=60)
ax.legend()

# Show the plot
plt.show()
# save it
plt.savefig("timing_comparisons_bar_graph.png")