import torch
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

def generate_predictions_and_features(model,images,batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if images.dtype == np.uint8:
        images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

    # build dataset
    dataset = TensorDataset(
        torch.from_numpy(images), 
        torch.from_numpy(np.ones(images.shape[0]))
        )

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # run inference 
    all_features = []
    all_predictions = []
    t0 = time.time()

    for k, (images, labels) in enumerate(dataloader):

        images = images.float().to(device)

        predictions, features = model.get_predictions_and_features(images)
        predictions = predictions.detach().cpu().numpy()
        features = features.detach().cpu().numpy().squeeze()

        all_predictions.append(predictions)
        all_features.append(features)

    predictions = np.vstack(all_predictions)
    features = np.vstack(all_features)

    print('running inference on ' + str(predictions.shape[0]) + 'images took ' + str(time.time()-t0) + ' s')

    # Plot
    print(predictions.shape)
    import matplotlib.pyplot as plt
    plt.clf()
    plt.hist(predictions, bins=100, range=[0,1])
    plt.xlabel('Output')
    plt.ylabel('Count')
    plt.title('Histogram of Outputs')
    plt.savefig('histograms_' + str(0)  + '.png', dpi=300)

    return predictions, features