'''Image ranking on tiny ImageNet with ResNet pre-trained on ImageNet.'''
from __future__ import print_function
import os
import argparse
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from sampler import triplet_sampler, emb_sampler, val_sampler
from dataSet import trainData, embData, valData


# parse command line input
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--batch_size', type=int, help='batch size for training')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--n_epoch', type=int, help='Num of epochs for training')
parser.add_argument('--resume_path', type=str, help='path to ckpt from which to resume training')
args = parser.parse_args()

# sanity check
if not args.model:
    print("Error: please specify --model!")
    raise SystemExit
else:
    print("Using {}".format(args.model))

if not args.batch_size:
    print("Error: please specify --batch_size!")
    raise SystemExit
else:
    print("Using batch size of {}".format(args.batch_size))

print("Learning rate is: {}".format(args.lr))

if not args.n_epoch:
    print("Error: please specify --n_epoch!")
    raise SystemExit
else:
    print("Running for {} epochs".format(args.n_epoch))

if args.resume_path:
    if not os.path.isfile(args.resume_path):
        print("Invalid ckpt path: {}".format(args.resume_path))
        raise SystemExit

# data augmentation
# normalization values are recommended by: https://pytorch.org/docs/stable/torchvision/models.html
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),  # resize to same size as ImageNet images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# load pretrained model, replace fc, the fine tune the entire network
model_urls = {
    'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

start_epoch = 0
best_loss = np.infty
if args.model == 'resnet18':
    net = models.resnet18()
elif args.model == 'resnet34':
    net = models.resnet34()
elif args.model == 'resnet50':
    net = models.resnet50()
elif args.model == 'resnet152':
    net = models.resnet152()
else:
    net = models.resnet101()

net.load_state_dict(model_zoo.load_url(model_urls[args.model], model_dir='./'))
n_featmap = net.fc.in_features
net.fc = nn.Linear(n_featmap, 4096)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

loss_fn = nn.TripletMarginLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.99)


def annotate_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return name

# driver code
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
ckpt_dir = 'resnet101'
ckpt_dir = annotate_dir(ckpt_dir)
ckpt_dir = os.path.join('./checkpoint', ckpt_dir)
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)


def train(epoch, trainloader):
    global best_loss

    net.train()
    train_loss = 0.0
    time1 = time.time()
    for batch_idx, (query_img, positive_img, negative_img) in enumerate(trainloader):
        query_img= query_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)
        optimizer.zero_grad()
        query_emb = net(query_img)
        positive_emb = net(positive_img)
        negative_emb = net(negative_img)

        loss = loss_fn(query_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    time2 = time.time()
    total_loss = train_loss / len(trainloader)
    sec = time2-time1
    min, sec = divmod(sec, 60)
    hr, min = divmod(min, 60)
    print('Epoch: {} | Train Loss: {:.3f} | Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(epoch, total_loss, hr, min, sec))

    if total_loss < best_loss:
        best_loss = total_loss
        print("Saving ckpt at {}-th epoch.".format(epoch))
        ckpt_name = 'model-' + str(epoch) + '.ckpt'
        state = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            'opt_state': optimizer.state_dict()
        }
        torch.save(state, os.path.join(ckpt_dir, ckpt_name))


indir = './tiny-imagenet-200/train'
if args.resume_path:
    print("Loading ckpt from: {}".format(args.resume_path))
    checkpoint = torch.load(args.resume_path)
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state'])
    print("ckpt loaded")

for epoch in range(start_epoch, args.n_epoch):

    triplet_arr = triplet_sampler(indir)
    trainset = trainData(triplet_arr, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    scheduler.step()  # adjust lr
    train(epoch, trainloader)

