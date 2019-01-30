import torch.utils.data as data
from PIL import Image
from sampler import triplet_sampler, val_sampler, emb_sampler


def img_loader(path):
    return Image.open(path).convert('RGB')


class trainData(data.Dataset):
    def __init__(self, triplet_list, transform=None, loader=img_loader):
        self.triplet_l = triplet_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        query_path, positive_path, negative_path= self.triplet_l[index]
        query_img = self.loader(query_path)
        positive_img = self.loader(positive_path)
        negative_img = self.loader(negative_path)
        if self.transform is not None:
            query_img = self.transform(query_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return query_img, positive_img, negative_img

    def __len__(self):
        return len(self.triplet_l)


class embData(data.Dataset):
    def __init__(self, emb_list, transform=None, loader=img_loader):
        self.emb_list = emb_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        query_path, query_label = self.emb_list[index]
        query_img = self.loader(query_path)
        if self.transform is not None:
            query_img = self.transform(query_img)

        return query_img, query_label, query_path

    def __len__(self):
        return len(self.emb_list)


class valData(data.Dataset):
    def __init__(self, val_list, transform=None, loader=img_loader):
        self.val_l = val_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        query_path, label = self.val_l[index]
        query_img = self.loader(query_path)
        if self.transform is not None:
            query_img = self.transform(query_img)

        return query_img, label, query_path

    def __len__(self):
        return len(self.val_l)


if __name__ == '__main__':
    indir = './tiny-imagenet-200/train'
    triplet_arr = triplet_sampler(indir)
    print(triplet_arr.shape)
    print("Training set:")
    train_dataset = trainData(triplet_arr)
    for i in range(3):
        query_img, positive_img, negative_img = train_dataset[i]
        print(query_img)
        print(positive_img)
        print(negative_img)


    emb_arr = emb_sampler(indir)
    print(emb_arr.shape)
    print("Embedding database:")
    emb_dataset = embData(emb_arr)
    for i in range(3):
        query_img, query_label, query_path = emb_dataset[i]
        print(query_img)
        print(query_label)
        print(query_path)


    val_path = './tiny-imagenet-200/val/images'
    val_f = './tiny-imagenet-200/val/val_annotations.txt'
    val_arr = val_sampler(val_path, val_f)
    print(val_arr.shape)
    print("Test set:")
    val_dataset = valData(val_arr)
    for i in range(3):
        img, label, query_path = val_dataset[i]
        print(img)
        print(label)
        print(query_path)

