"""Train CIFAR100 with PyTorch."""
from __future__ import print_function

import argparse
import os

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import PreActResNet18
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--exp', default='cifar100_mixup', type=str,
                    help='name of the experiment')
parser.add_argument('--mixup', action='store_true',
                    help='whether to use mixup or not')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='/data/public/cifar100', train=True, download=True,
    transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='/data/public/cifar100', train=False, download=True,
    transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_{}/ckpt.t7'.format(args.exp))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = PreActResNet18(num_classes=100)
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

# Training


def shuffle_minibatch(inputs, targets, mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.

    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
    batch_size = inputs.shape[0]

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 100)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 100)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = numpy.random.beta(1, 1, [batch_size, 1])
    else:
        a = numpy.ones((batch_size, 1))

    b = numpy.tile(a[..., None, None], [1, 3, 32, 32])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = numpy.tile(a, [1, 100])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle


def train(epoch):
    """Training function."""
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs_shuffle, targets_shuffle = shuffle_minibatch(
            inputs, targets, args.mixup)

        if use_cuda:
            inputs_shuffle, targets_shuffle = inputs_shuffle.cuda(), \
                targets_shuffle.cuda()

        optimizer.zero_grad()

        inputs_shuffle, targets_shuffle = Variable(
            inputs_shuffle), Variable(targets_shuffle)

        outputs = net(inputs_shuffle)
        m = nn.LogSoftmax()

        loss = -m(outputs) * targets_shuffle
        loss = torch.sum(loss) / 128
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        _, targets = torch.max(targets_shuffle.data, 1)
        correct += predicted.eq(targets).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Epoch %d, Training Loss: %.3f | Acc: %.3f%% (%d/%d)'  # noqa
                     % (epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))  # noqa


def test(epoch):
    """Testing function."""
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Epoch %d, Test Loss: %.3f | Acc: %.3f%% (%d/%d)'  # noqa
                     % (epoch, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))  # noqa

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_{}'.format(args.exp)):
            os.mkdir('checkpoint_{}'.format(args.exp))
        torch.save(state, './checkpoint_{}/ckpt.t7'.format(args.exp))
        best_acc = acc

scheduler = lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], gamma=0.1)
for epoch in range(start_epoch, start_epoch + 200):
    scheduler.step()
    train(epoch)
    test(epoch)
