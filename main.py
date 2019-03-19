import argparse
import sys
import torch
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
batch_size = 8
n_epochs = 150
GPU_ids = 0
nclass = 2
momentum = 0.9
nesterov = True
weight_decay = 0.0005
milestones = [50, 75, 120]
gamma = 0.1
modelSaveFn = 'myModel'

parser = argparse.ArgumentParser(description='MyFirstCNNTry')
parser.add_argument('--lr', default=lr)
parser.add_argument('--batch_size', default=batch_size)
parser.add_argument('--n_epochs', default=n_epochs)
parser.add_argument('--GPU_ids', default=GPU_ids)
parser.add_argument('--nclass', default=nclass)
parser.add_argument('--momentum', default=momentum)
parser.add_argument('--nesterov', default=nesterov)
parser.add_argument('--weight_decay', default=weight_decay)
parser.add_argument('--milestones', default=milestones)
parser.add_argument('--gamma', default=gamma)
parser.add_argument('--modelSaveFn', default=modelSaveFn)
args = parser.parse_args()


model_name = 'densenet169'
args.model_name = model_name
if model_name == 'resnet152':
    original_model = torchvision.models.resnet152(pretrained=True)
elif model_name == 'resnet50':
    original_model = torchvision.models.resnet50(pretrained=True)
elif model_name == 'densenet121':
    original_model = torchvision.models.densenet121(pretrained=True)
elif model_name == 'densenet161':
    original_model = torchvision.models.densenet161(pretrained=True)
elif model_name == 'densenet169':
    original_model = torchvision.models.densenet169(pretrained=True)
else:
    original_model = ""
    print("Invalid model name, exiting...")
    exit()

# for name, param in original_model.named_children():
#     # print("\nOriginal Model => \n", name, "-------->", param)
#     print("\nOriginal Model => \n", name)
#
net = fineTuneModel(original_model, model_name, args)
# for name, param in net.named_children():
#     # print("\nFine-Tuned Model => \n", name, "-------->", param)
#     print("\nFine-Tuned Model => \n", name)

# Details
f = open("./results/params.txt", "w+")
f.write('Model Name: %s\n' % model_name)
f.write('Learning Rate: %.6f\n' % lr)
f.write('Batch Size: %3d\n' % batch_size)
f.write('Epochs: %3d\n' % n_epochs)
f.write('Classes: %2d\n' % nclass)
f.write('momentum: %.3f\n' % momentum)
f.write('nesterov: %s\n' % True)
f.write('weight_decay: %.6f\n' % weight_decay)
f.write('milestones: [ ')
for num in milestones:
    f.write('%d ' % num)
f.write(']\n')
f.write('gamma: %.3f\n' % gamma)
f.close()

cnn = cnnTrain(net, args)
max_valid_kappa = cnn.iterateCNN()

print('\nStart time: ', time.ctime(int(start)))
# End time
end = time.time()
print('End time: ', time.ctime(int(time.time())))
time = float(end - start)
day = time // (24 * 3600)
time = time % (24 * 3600)
hour = time // 3600
time %= 3600
minutes = time // 60
time %= 60
seconds = time

# Details
print('\n----------- Parameter Details -------------')
print('Model Name: ', model_name)
print('Learning Rate: ', lr)
print('Batch Size: ', batch_size)
print('Epochs: ', n_epochs)
print('Classes: ', nclass)
print('momentum: ', momentum)
print('nesterov: ', True)
print('weight_decay: ', weight_decay)
print('milestones: ', milestones)
print('gamma: ', gamma)
print('GPU Name: ', torch.cuda.get_device_name(GPU_ids))
print('\nmax_valid_kappa: ', max_valid_kappa)
print('Execution time: ', end - start, '\nExecution time (d:h:m:s): %d:%d:%d:%d' % (day, hour, minutes, seconds))
