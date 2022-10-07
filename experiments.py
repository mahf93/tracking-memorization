# code adapted from https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"



import argparse
import pickle

from models import *
from dataset import *

def train(args, epoch, trainloader, net, optimizer, device, criterion):
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_loss_clean = 0
    correct_clean = 0
    train_loss_noisy = 0
    correct_noisy = 0
    total_clean = 0
    total_noisy = 0
    
    for batch_idx, (inputs, targets, or_targets) in enumerate(trainloader):
        if device == 'cuda':
            inputs, targets, or_targets = inputs.cuda(), targets.cuda(), or_targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted.eq(targets.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # compute train loss on the noisy and clean subset:
        for tar, or_tar, ou in zip(targets, or_targets, outputs):
            tar = tar.unsqueeze(0)
            or_tar = or_tar.unsqueeze(0)
            ou = ou.unsqueeze(0)
            loss = criterion(ou, tar)
            _, pred = ou.max(1)
            cor = pred.eq(tar).sum().item()
            tot = tar.size(0)
            if tar.item() == or_tar.item():
                train_loss_clean += loss.item()
                correct_clean += cor
                total_clean += tot
            else:
                train_loss_noisy += loss.item()
                correct_noisy += cor
                total_noisy += tot
    if args.corruptprob == 0:
        total_noisy = 1 

    return train_loss/(batch_idx+1), 100.*correct/total,train_loss_clean/(total_clean), 100.*correct_clean/total_clean,train_loss_noisy/(total_noisy), 100.*correct_noisy/total_noisy

def test(args, epoch, testloader, net, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            
    return test_loss/(batch_idx+1), 100.*correct/total

def compute_update_on_sout(args, epoch, noisytrainloader, net, modelfilename, optimizer, device, criterion):
    
    # save the model here
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, modelfilename)
    
    net.train()
    counter = 0
    
    it = iter(noisytrainloader)
    
    inputs, targets, or_targets = next(it)

    if device == 'cuda':
        inputs, targets = inputs.cuda(), targets.cuda()

    # take one step on Sout
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)

    # \tilde{Phi}(w(t))
    train_loss_bef = loss.item()

    loss.backward()
    optimizer.step()

    # \tilde{Phi}(\tilde{w}(t+1))
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    train_loss_aft = loss.item()
                
    # before going out from here load the model again here so that training is not affected!
    checkpoint = torch.load(modelfilename)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['opt'])
    
    
    return train_loss_bef, train_loss_aft

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Tracking Memorization Code")
    parser.add_argument("--model", type=str, choices=['cnn','densenet','efficientnet','googlenet','mobilenet','mobilenetv2','resnet','resnext','senet','shufflenetv2','vgg'], help="the neural network configuration")
    parser.add_argument("--scale", type=float, default=1, help="the scale to use for the configuration")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset choices are cifar10")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes in the dataset")
    parser.add_argument("--numsamples", type=int, default=50000, help="number of training samples to train on")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size of both the training and the testing sets")
    parser.add_argument("--corruptprob", type=float, default=0.5, help="the corrupt probability of the labels of the training samples")
    parser.add_argument("--epochs", type=int, default=200, help="maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="sgd learning rate")
    parser.add_argument("--filename", type=str, default='', help="filename to save the results to")
    parser.add_argument("--modelfilename", type=str, default='', help="filename to save the model on")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    torch.cuda.manual_seed(args.seed)
    print('==> Building model..')
    if args.model == 'cnn':
        net = make_cnn(num_planes=args.scale, num_classes = args.num_classes)
    elif args.model == 'densenet':
        net = DenseNet121(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'efficientnet':
        net = EfficientNetB0(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'googlenet':
        net = GoogLeNet(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'mobilenet':
        net = MobileNet(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'mobilenetv2':
        net = MobileNetV2(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'resnet':
        net = ResNet18(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'resnext':
        net = ResNeXt29_2x64d(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'senet':
        net = SENet18(num_classes = args.num_classes, scale=args.scale)
    elif args.model == 'shufflenetv2':
        net = ShuffleNetV2(net_size=args.scale, num_classes = args.num_classes)
    elif args.model == 'vgg':
        net = VGG('VGG19', num_classes = args.num_classes, scale=args.scale)
    
    net = net.to(device)
    
    print('==> Building dataloader..')
    train_loader = get_data_loader(batch_size=args.batchsize, train=True, num_samples=args.numsamples, corrupt_prob = args.corruptprob)
    noisy_train_loader = get_data_loader(batch_size=args.batchsize, train=True, num_samples=args.numsamples, corrupt_prob = 1)
    test_loader = get_data_loader(batch_size=args.batchsize, train=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    file_name = './results/{}_{}_{}_{}.pkl'.\
    format(args.filename, args.model, args.epochs, args.lr)
    model_file_name = './checkpoint/{}_{}_{}_{}.pkl'.\
    format(args.modelfilename, args.model, args.epochs, args.lr)
    
    print('==> Start training..')
    
    with open(file_name, 'wb') as f:
        pickler = pickle.Pickler(f)
        for epoch in range(0, args.epochs):
            print('Epoch:', epoch)
            
            phi_out_bef, phi_out_aft = compute_update_on_sout(args, epoch, noisy_train_loader, net, model_file_name, optimizer, device, criterion)
        
        
            tl, ta, tlc, tac, tln, tan = train(args, epoch, train_loader, net, optimizer, device, criterion)
            tel, tea = test(args, epoch, test_loader, net, device, criterion)
            scheduler.step()
            pickler.dump([epoch, phi_out_bef, phi_out_aft,
                         tl, ta, tlc, tac, tln, tan,
                         tel, tea])
    print('==> Training is done.')

if __name__ == "__main__":
    main()


