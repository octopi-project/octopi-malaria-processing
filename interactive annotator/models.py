import torch
import torch.nn as nn
import torchvision.models

models_dict = {'resnet18': models.resnet18,
               'resnet34': models.resnet34,
               'resnet50': models.resnet50,
               'resnet101': models.resnet101,
               'resnet152': models.resnet152}

class ResNet(nn.Module):
    def __init__(self, model='resnet18',n_channels=4,n_filters=64,n_classes=1,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.n_classes = n_classes
        self.base_model = models_dict[model]()
        self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
        self.fc = nn.Linear(self.base_model.fc.in_features, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        return self.fc(features)

    def _feature_vector_dimension():
        return self.base_model.fc.in_features

    def extract_features(self,x):
        x = self.base_model(x)
        return x.view(x.size(0), -1)

    def get_predictions(self,x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        output = self.fc(features)
        if self.n_classes == 1:
            return torch.sigmoid(output)
        else:
            return torch.softmax(output,dim=1)

    def get_predictions_and_features:
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        output = self.fc(features)
        if self.n_classes == 1:
            return torch.sigmoid(output), features
        else:
            return torch.softmax(output,dim=1), features