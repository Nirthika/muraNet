import torch.nn as nn
import torch.nn.functional as F

class fineTuneModel(nn.Module):
    def __init__(self, original_model, arch, args):
        super(fineTuneModel, self).__init__()
        
        if arch.startswith('densenet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            num_ftrs = original_model.classifier.in_features
            self.classifier = nn.Sequential(nn.Linear(num_ftrs, args.nclass), nn.Sigmoid())
            self.modelName = 'densenet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            num_ftrs = original_model.fc.in_features
            self.classifier = nn.Sequential(nn.Linear(num_ftrs, args.nclass), nn.Softmax())
            self.modelName = 'resnet'
        else:
            raise("Finetuning not supported on this architecture yet")

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
