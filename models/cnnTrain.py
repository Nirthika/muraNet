from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.getData import getData
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

class cnnTrain(nn.Module):
    def __init__(self, net, args):
        super(cnnTrain, self).__init__()
        self.args = args
        if torch.cuda.is_available():
            self.net = net.cuda(self.args.GPU_ids)

        self.trainloader, self.validloader, self.ntrain, self.nvalid = self.get_loaders()

        # Loss and Optimizer
        tr_weight = [args.weights[0], args.weights[1]]
        self.class_weights = torch.FloatTensor(tr_weight).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, size_average=False).cuda(self.args.GPU_ids)

        # self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.005)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             mode='min',
        #                                                             patience=5,
        #                                                             verbose=True)

        self.optimizer = optim.SGD([{'params': net.features.parameters(), 'lr': args.lr},
                                    {'params': net.classifier.parameters(), 'lr': args.lr * 10}],
                                   lr=args.lr,
                                   momentum=args.momentum,
                                   nesterov=args.nesterov,
                                   weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=args.milestones,
                                                              gamma=args.gamma)

        self.print_net()

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        classArr = []
        target = []
        predicted = []
        print('Training----->')
        for batch_idx, (inputs, targets, classes) in enumerate(self.trainloader):
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
            target.extend(targets.cpu().numpy())
            predicted.extend(pred.cpu().numpy())
            for i in range(len(classes)):
                classArr.append(classes[i])

            if (batch_idx + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (
                    epoch + 1, self.args.n_epochs, batch_idx + 1, self.ntrain // self.args.batch_size, train_loss))

        acc, mca = self.getMCA(target, predicted)
        # kappa_p, kappa_n = self.getBinaryKappa(target, predicted)
        O_kappa = cohen_kappa_score(target, predicted)
        return train_loss, acc, mca, O_kappa, target, predicted, classArr

    def valid(self, epoch):
        self.net.eval()
        valid_loss = 0
        classArr = []
        target = []
        predicted = []
        print('Validating----->')
        for batch_idx, (inputs, targets, classes) in enumerate(self.validloader):
            inputs, targets = inputs.cuda(self.args.GPU_ids), targets.cuda(self.args.GPU_ids)

            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

            valid_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            target.extend(targets.cpu().numpy())
            predicted.extend(pred.cpu().numpy())
            for i in range(len(classes)):
                classArr.append(classes[i])

            if (batch_idx + 1) % 100 == 0:
                print('Completed: [%d/%d]' % (batch_idx + 1, self.nvalid // self.args.batch_size))

        acc, mca = self.getMCA(target, predicted)
        # kappa_p, kappa_n = self.getBinaryKappa(target, predicted)
        O_kappa = cohen_kappa_score(target, predicted)
        return valid_loss, acc, mca, O_kappa, target, predicted, classArr

    def print_net(self):
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
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        transform_valid = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        # Dataset
        print('\nPreparing data----->')
        trainset = getData(self.args, train=True, transform=transform_train)
        validset = getData(self.args, train=False, transform=transform_valid)

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
        outputs = []
        for epoch in range(self.args.n_epochs):
            self.scheduler.step()
            tr_loss, tr_acc, tr_mca, tr_kappa_o, tr_target, tr_pred, tr_cla = self.train(epoch)
            va_loss, va_acc, va_mca, va_kappa_o, va_target, va_pred, va_cla = self.valid(epoch)

            outputs.append(
                [tr_loss, tr_acc, tr_mca, tr_kappa_o, va_loss, va_acc, va_mca, va_kappa_o])

            print('----------------------')
            print('Epoch\tTrLoss\t\tTrAcc\t\tTrMCA\t\tOKappaTr\tVaLoss\t\tVaAcc\t\tVaMCA\t\tOKappaVa')
            for i in range(len(outputs)):
                print('%1d \t %.4f \t %.3f%% \t %.3f%% \t %.3f \t\t %.4f \t %.3f%% \t %.3f%% \t\t %.3f' %
                    (i, outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3], outputs[i][4], outputs[i][5],
                     outputs[i][6], outputs[i][7]))
                # if i == (self.args.n_epochs-1):
                #     print("\nTraining Kappa ----->")
                #     self.getClassKappa(tr_target, tr_pred, tr_cla)
                #     print("Validating Kappa ----->")
                #     self.getClassKappa(va_target, va_pred, va_cla)
            if epoch == (self.args.n_epochs-1):
                print("\nTraining Kappa ----->")
            self.getClassKappa(tr_target, tr_pred, tr_cla, epoch)
            if epoch == (self.args.n_epochs-1):
                print("Validating Kappa ----->")
            self.getClassKappa(va_target, va_pred, va_cla, epoch)

        f = open("./results/lastResults.txt", "w+")
        f.write('Epoch\tTrLoss\t\tTrAcc\t\tTrMCA\t\tOKappaTr\tVaLoss\t\tVaAcc\t\tVaMCA\t\tOKappaVa')
        for i in range(len(outputs)):
            f.write('%1d \t %.4f \t %.3f%% \t %.3f%% \t %.3f \t\t %.4f \t %.3f%% \t %.3f%% \t\t %.3f' %
                    (i, outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3], outputs[i][4], outputs[i][5],
                     outputs[i][6], outputs[i][7]))
        f.close()

        torch.save(self.net, self.args.modelSaveFn)
        max_valid_kappa = max([_[7] for _ in outputs])

        # Plot Epoch Vs TrLoss
        ep = [i for i in range(self.args.n_epochs)]
        tl = [i[0] for i in outputs]
        plt.figure(0)
        plt.plot(ep, tl)
        plt.suptitle('Epoch Vs TrLoss')
        plt.xlabel('Epoch')
        plt.ylabel('TrLoss')
        plt.savefig('EpochVsTrLoss.png')

        # Plot Epoch Vs VaLoss
        vl = [i[4] for i in outputs]
        plt.figure(1)
        plt.plot(ep, vl)
        plt.suptitle('Epoch Vs VaLoss')
        plt.xlabel('Epoch')
        plt.ylabel('VaLoss')
        plt.savefig('EpochVsVaLoss.png')

        # Plot Epoch Vs KappaTr and KappaVa
        tk = [i[3] for i in outputs]
        vk = [i[7] for i in outputs]
        plt.figure(2)
        plt.plot(ep, tk)
        plt.plot(ep, vk)
        plt.suptitle('Epoch Vs Kappa')
        plt.xlabel('Epoch')
        plt.ylabel('Kappa')
        plt.legend(['KappaTr', 'KappaVa'], loc='upper left')
        plt.savefig('EpochVsKappa.png')

        return max_valid_kappa

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

    def getBinaryKappa(self, target_lbls, predicted_lbls):
        target_p, predicted_p, target_n, predicted_n = [], [], [], []
        for i in range(len(self.class_weights)):
            for j, x in enumerate(target_lbls):
                if x == 0:
                    target_n.append(x)
                    predicted_n.append(predicted_lbls[j])
                elif x == 1:
                    target_p.append(x)
                    predicted_p.append(predicted_lbls[j])

        kappa_p = cohen_kappa_score(target_p, predicted_p)
        kappa_n = cohen_kappa_score(target_n, predicted_n)
        return kappa_p, kappa_n

    def getClassKappa(self, target_lbls, predicted_lbls, class_lbls, epoch):
        # type = ['SHOULDER', 'HUMERUS', 'FINGER', 'ELBOW', 'WRIST', 'FOREARM', 'HAND']
        shoulder_tar, shoulder_pre = [], []
        humerus_tar, humerus_pre = [], []
        finger_tar, finger_pre = [], []
        elbow_tar, elbow_pre = [], []
        wrist_tar, wrist_pre = [], []
        forearm_tar, forearm_pre = [], []
        hand_tar, hand_pre = [], []

        shoulder_ka, humerus_ka, finger_ka, elbow_ka, wrist_ka, forearm_ka, hand_ka = [], [], [], [], [], [], []

        for i, x in enumerate(target_lbls):
            if class_lbls[i] == 'SHOULDER':
                shoulder_tar.append(x)
                shoulder_pre.append(predicted_lbls[i])
            elif class_lbls[i] == 'HUMERUS':
                humerus_tar.append(x)
                humerus_pre.append(predicted_lbls[i])
            elif class_lbls[i] == 'FINGER':
                finger_tar.append(x)
                finger_pre.append(predicted_lbls[i])
            elif class_lbls[i] == 'ELBOW':
                elbow_tar.append(x)
                elbow_pre.append(predicted_lbls[i])
            elif class_lbls[i] == 'WRIST':
                wrist_tar.append(x)
                wrist_pre.append(predicted_lbls[i])
            elif class_lbls[i] == 'FOREARM':
                forearm_tar.append(x)
                forearm_pre.append(predicted_lbls[i])
            elif class_lbls[i] == 'HAND':
                hand_tar.append(x)
                hand_pre.append(predicted_lbls[i])

        shoulder_ka.append(cohen_kappa_score(shoulder_tar, shoulder_pre))
        humerus_ka.append(cohen_kappa_score(humerus_tar, humerus_pre))
        finger_ka.append(cohen_kappa_score(finger_tar, finger_pre))
        elbow_ka.append(cohen_kappa_score(elbow_tar, elbow_pre))
        wrist_ka.append(cohen_kappa_score(wrist_tar, wrist_pre))
        forearm_ka.append(cohen_kappa_score(forearm_tar, forearm_pre))
        hand_ka.append(cohen_kappa_score(hand_tar, hand_pre))

        if epoch == (self.args.n_epochs-1):
            print('Class\t\tKappa')
            print('SHOULDER:\t%.3f' % max(shoulder_ka))
            print('HUMERUS: \t%.3f' % max(humerus_ka))
            print('FINGER:\t\t%.3f' % max(finger_ka))
            print('ELBOW: \t\t%.3f' % max(elbow_ka))
            print('WRIST: \t\t%.3f' % max(wrist_ka))
            print('FOREARM: \t%.3f' % max(forearm_ka))
            print('HAND:\t\t%.3f' % max(hand_ka))
            print('------------------------------------------------')
