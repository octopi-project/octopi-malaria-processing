import numpy as np
import pandas as pd

# data_dir = '/home/rinni/Desktop/Octopi/data/unsure-image-training'
# dataset_id = ['7_unsure_negative_images_2023-04-13_18-46', 'PAT-071-3_2023-01-22_15-47-3.096602_definitely_positive_images_2023-04-13_18-50', 'PAT-073-1_2023-01-22_16-32-5.192404_unsure_positive_images_2023-04-13_18-37']
# annotation_dict = {dataset_id[0]:'7_unsure_negative_images_2023-04-13_18-46_annotations', dataset_id[1]:'PAT-071-3_2023-01-22_15-47-3.096602_definitely_positive_images_2023-04-13_18-50_annotations', dataset_id[2]:'PAT-073-1_2023-01-22_16-32-5.192404_unsure_positive_images_2023-04-13_18-37_annotations'}

##########################################################################
config_files = glob.glob('.' + '/' + 'def*.txt')
if config_files:
    if len(config_files) > 1:
        print('multiple def files found, the program will exit')
        exit()
    exec(open(config_files[0]).read())
##########################################################################

images = []
annotations = []

for id_ in dataset_id:

	images_ = np.load(data_dir + '/' + id_ + '.npy')

	if id_ in annotation_dict.keys():
		annotation_pd = pd.read_csv(data_dir + '/' + annotation_dict[id_] + '.csv',index_col='index')
		annotation_pd = annotation_pd.sort_index()
		idx = annotation_pd['annotation'].isin([0, 1])
		images_ = images_[idx,]
		annotation_pd = annotation_pd[idx]
		annotations_ = annotation_pd['annotation'].values.squeeze()
	else:
		# sample
		random_indexes = np.random.randint(0, images_.shape[0], size=min(images_.shape[0],100000))
		images_ = images_[random_indexes,]
		annotations_ = np.zeros(len(images_))

	print(images_.shape)
	images.append(images_)
	annotations.append(annotations_)

images = np.concatenate(images,axis=0)
annotations = np.concatenate(annotations)

print(images.shape)

np.save(data_dir + '/' + str(dataset_id) + ' .npy',images)
# np.save(data_dir + '/combined_unsure.npy',images)
df = pd.DataFrame({'annotation':annotations})
df.index.name = 'index'
df.to_csv(data_dir + '/' + str(dataset_id) + '.csv')
# df.to_csv(data_dir + '/combined_unsure.csv')