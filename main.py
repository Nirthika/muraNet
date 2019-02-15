import argparse
import sys
sys.path.append('./models/')
from models.fineTune import fineTuneModel
from models.cnnTrain import cnnTrain
import torchvision
import time

# Start time
start = time.time()
print('Start time: ', time.ctime(int(time.time())), '\n')

# parameters
lr = 0.0001
batch_size = 40
n_epochs = 100
GPU_ids = 0
nclass = 2
modelSaveFn = 'myModel'

parser = argparse.ArgumentParser(description='MyFirstCNNTry')
parser.add_argument('--lr', default=lr)
parser.add_argument('--batch_size', default=batch_size)
parser.add_argument('--n_epochs', default=n_epochs)
parser.add_argument('--GPU_ids', default=GPU_ids)
parser.add_argument('--modelSaveFn', default=modelSaveFn)
args = parser.parse_args()


model_name = 'densenet169'
args.model_name = model_name
if model_name == 'resnet152':
    original_model = torchvision.models.resnet152(pretrained=True)
elif model_name == 'resnet50':
    original_model = torchvision.models.resnet50(pretrained=True)
elif model_name == 'densenet169':
    original_model = torchvision.models.densenet169(pretrained=True)
elif model_name == 'densenet121':
    original_model = torchvision.models.densenet121(pretrained=True)
else:
    original_model = ""
    print("Invalid model name, exiting...")
    exit()

# for name, param in original_model.named_children():
#     # print("\nOriginal Model => \n", name, "-------->", param)
#     print("\nOriginal Model => \n", name)
#
net = fineTuneModel(original_model, model_name)
# for name, param in net.named_children():
#     # print("\nFine-Tuned Model => \n", name, "-------->", param)
#     print("\nFine-Tuned Model => \n", name)


cnn = cnnTrain(net, args)
mca = cnn.iterateCNN()

# End time
end = time.time()
print('\n', 'End time: ', time.ctime(int(time.time())))
print('Execution time: ', end - start, '\n')
