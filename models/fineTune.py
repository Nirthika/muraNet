import torch.nn as nn

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
            self.classifier = nn.Sequential(nn.Linear(num_ftrs, args.nclass))
            self.modelName = 'resnet'
        else :
            raise("Finetuning not supported on this architecture yet")

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
