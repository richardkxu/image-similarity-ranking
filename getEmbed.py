from __future__ import print_function
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from sampler import triplet_sampler, emb_sampler, val_sampler
from dataSet import trainData, embData, valData


# parse command line input
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--batch_size', type=int, help='batch size for embedding loader')
parser.add_argument('--ckpt_path', type=str, help='path to ckpt')
args = parser.parse_args()


# no data aug for test set
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def compute_emb(emb_loader):

    net.eval()
    with torch.no_grad():
        database_embs = np.zeros((args.batch_size*len(emb_loader), 4096))
        database_labels = []
        database_paths = []
        for batch_idx, (query_img, query_label, query_path) in enumerate(emb_loader):
            query_img = query_img.to(device)
            query_emb = net(query_img)
            start = batch_idx*args.batch_size
            database_embs[start:start + args.batch_size] = query_emb.cpu().numpy()
            database_labels = database_labels + list(query_label)
            database_paths = database_paths + list(query_path)

    return database_embs, database_labels, database_paths


# load model and ckpt
model_urls = {
    'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

checkpoint_path = ''
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
n_featmap = net.fc.in_features
net.fc = nn.Linear(n_featmap, 4096)

print("Loading ckpt from: {}".format(args.ckpt_path))
checkpoint = torch.load(args.ckpt_path)
net.load_state_dict(checkpoint['state_dict'])
print("ckpt loaded")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# sample training database
train_path = '/data/kexu6/tiny-imagenet-200/train'
emb_arr = emb_sampler(train_path)
print(emb_arr.shape)
embset = embData(emb_arr, transform=transform_test)
embloader = torch.utils.data.DataLoader(embset, batch_size=args.batch_size, shuffle=False, num_workers=16)

# sample test database
val_path = '/data/kexu6/tiny-imagenet-200/val/images'
val_f = '/data/kexu6/tiny-imagenet-200/val/val_annotations.txt'
val_arr = val_sampler(val_path, val_f)
print(val_arr.shape)
valset = valData(val_arr, transform=transform_test)
testloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=16)

# compute training embedding
time1 = time.time()
train_embs, train_labels, train_paths = compute_emb(embloader)
time2 = time.time()
sec = time2-time1
min, sec = divmod(sec, 60)
hr, min = divmod(min, 60)
print('Training emb time: {:.2f} hr {:.2f} min {:.2f} sec'.format(hr, min, sec))
train_labels = np.array(train_labels)
train_paths = np.array(train_paths)
print("Train embs shape: {}".format(train_embs.shape))
print("Train labels shape: {}".format(train_labels.shape))
print("Train paths shape: {}".format(train_paths.shape))
np.save('./train_embs', arr=train_embs)
np.save('./train_labels', arr=train_labels)
np.save('./train_paths', arr=train_paths)


# compute test embedding
time3 = time.time()
test_embs, test_labels, test_paths = compute_emb(testloader)
time4 = time.time()
sec = time4-time3
min, sec = divmod(sec, 60)
hr, min = divmod(min, 60)
print('Test emb time: {:.2f} hr {:.2f} min {:.2f} sec'.format(hr, min, sec))
test_labels = np.array(test_labels)
test_paths = np.array(test_paths)
print("Test embs shape: {}".format(test_embs.shape))
print("Test labels shape: {}".format(test_labels.shape))
print("Test paths shape: {}".format(test_paths.shape))
np.save('./test_embs', arr=test_embs)
np.save('./test_labels', arr=test_labels)
np.save('./test_paths', arr=test_paths)

