from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.getData import getData
from sklearn.metrics import cohen_kappa_score


class cnnTrain(nn.Module):
    def __init__(self, net, args):
        super(cnnTrain, self).__init__()
        self.args = args
        if torch.cuda.is_available():
            self.net = net.cuda(self.args.GPU_ids)

        self.trainloader, self.validloader, self.ntrain, self.nvalid = self.get_loaders()

        # Loss and Optimizer
        weights = [0.404, 0.596]
        self.class_weights = torch.FloatTensor(weights).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).cuda(self.args.GPU_ids)

        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    patience=5,
                                                                    verbose=True)
        # self.optimizer = optim.SGD([{'params': net.features.parameters(), 'lr': args.lr},
        #                             {'params': net.classifier.parameters(), 'lr': args.lr*10}],
        #                            lr=args.lr,
        #                            momentum=0.9,
        #                            nesterov=True,
        #                            weight_decay=0.0005)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                       milestones=[200, 300],
        #                                                       gamma=0.1)

        self.print_net()

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = []
        predicted = []
        print('Training----->')
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # transfer the inputs to the GPU, and set them variables
            inputs, targets = inputs.cuda(self.args.GPU_ids), targets.cuda(self.args.GPU_ids)
            inputs, targets = Variable(inputs), Variable(targets)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            correct.extend(targets.cpu().numpy())
            predicted.extend(pred.cpu().numpy())

            # print statistics
            if (batch_idx + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (
                      epoch + 1, self.args.n_epochs, batch_idx + 1, self.ntrain // self.args.batch_size, loss.item()))

        acc, mca = self.getMCA(correct, predicted)
        kappa = cohen_kappa_score(correct, predicted)
        return train_loss, acc, mca, kappa

    def valid(self, epoch):
        self.net.eval()
        valid_loss = 0
        correct = []
        predicted = []
        print('Validating----->')
        for batch_idx, (inputs, targets) in enumerate(self.validloader):
            inputs, targets = inputs.cuda(self.args.GPU_ids), targets.cuda(self.args.GPU_ids)
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

            valid_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            correct.extend(targets.cpu().numpy())
            predicted.extend(pred.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                print('Completed: [%d/%d]' % (batch_idx + 1, self.nvalid // self.args.batch_size))

        acc, mca = self.getMCA(correct, predicted)
        kappa = cohen_kappa_score(correct, predicted)
        return valid_loss, acc, mca, kappa

    def print_net(self):
        # print('----------------------------')
        # print(self.net)
        params = list(self.net.parameters())
        # for p in params:
        #    print(p.size())  # conv1's .weight
        pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(len(params))
        print('Total parameters %d' % (pytorch_total_params))
        pytorch_total_params = float(pytorch_total_params) / 10 ** 6
        print('Total parameters requires_grad %.3f M' % (pytorch_total_params))

        print('----------------------------')

    def get_loaders(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        imsize = 320

        transform_train = transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.RandomRotation(30),
            # transforms.RandomAffine([-10, 10], translate=[0.05, 0.05], scale=[0.7, 1.3]),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ])

        transform_valid = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ])

        # Dataset
        print('\nPreparing data----->')
        trainset = getData(train=True, transform=transform_train)
        validset = getData(train=False, transform=transform_valid)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.args.batch_size,
                                                  num_workers=5,
                                                  shuffle=True)
        validloader = torch.utils.data.DataLoader(validset,
                                                  batch_size=self.args.batch_size,
                                                  num_workers=1,
                                                  shuffle=False)
        return trainloader, validloader, len(trainset), len(validset)

    def iterateCNN(self):
        tr_loss_arr = []
        for epoch in range(self.args.n_epochs):
            # self.scheduler.step()
            train_loss, accTr, mcaTr, kappaTr = self.train(epoch)
            if epoch % 10 == 0:
                valid_loss, accVa, mcaVa, kappaVa = self.valid(epoch)
                self.scheduler.step(valid_loss)
            else:
                valid_loss, accVa, mcaVa, kappaVa = 0, 0, 0, 0
                self.scheduler.step(valid_loss)
            tr_loss_arr.append([train_loss, accTr, mcaTr, kappaTr, valid_loss, accVa, mcaVa, kappaVa])

            print('----------------------')
            print('Epoch\tTrLoss\t\tTrAcc\t\tTrMCA\t\tKappaTr\t\tVaLoss\t\tVaAcc\t\tVaMCA\t\tKappaVa');
            for i in range(len(tr_loss_arr)):
                print('%d \t %.4f \t %.3f%% \t %.3f%% \t %.3f \t\t %.4f \t %.3f%% \t %.3f%% \t %.3f' %
                      (i, tr_loss_arr[i][0], tr_loss_arr[i][1], tr_loss_arr[i][2], tr_loss_arr[i][3],
                       tr_loss_arr[i][4], tr_loss_arr[i][5], tr_loss_arr[i][6], tr_loss_arr[i][7]))
        mca = tr_loss_arr[-1][-1]

        torch.save(self.net, self.args.modelSaveFn)
        return mca

    def getMCA(self, correct_lbls, predicted_lbls):
        mca = 0
        acc = 0
        for lbl, w in enumerate(self.class_weights):
            correct_c = 0.0
            tot_c = 0.0
            for i, x in enumerate(correct_lbls):
                if x == lbl:
                    tot_c = tot_c + 1
                    if x == predicted_lbls[i]:
                        correct_c = correct_c + 1
            acc = acc + correct_c
            acc_t = correct_c / tot_c * 100.0
            mca = mca + acc_t
        mca = mca / len(self.class_weights)
        acc = acc / len(predicted_lbls) * 100
        return acc, mca
