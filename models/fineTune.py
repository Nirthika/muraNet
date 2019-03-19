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
        else :
            raise("Finetuning not supported on this architecture yet")

    def forward(self, x):
        f = self.features(x)
        # print(f.size(0), f.size(1), f.size(2))
        # f = f.view(f.size(0), -1)
        f = F.relu(f, inplace=True)
        f = F.avg_pool2d(f, kernel_size=7, stride=1).view(f.size(0), -1)
        y = self.classifier(f)
        return y
