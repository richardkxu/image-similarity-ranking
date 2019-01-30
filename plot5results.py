import os

import matplotlib.pyplot as plt
from PIL import Image

def img_loader(path):
    return Image.open(path).convert('RGB')

test_img = '/home/richardkxu/Downloads/tiny-imagenet-200/val/images/val_38.JPEG'

top10_imgs = \
[
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_209.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_61.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_287.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_357.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_27.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_62.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_461.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_177.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_492.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n04099969/images/n04099969_141.JPEG'
]

top10_labels = \
[
 'n04099969', 'n04099969', 'n04099969', 'n04099969', 'n04099969', 'n04099969',
 'n04099969', 'n04099969', 'n04099969', 'n04099969'
]

bottom10_imgs = \
[
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_88.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_57.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_438.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_292.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_352.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_39.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_282.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_463.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_262.JPEG',
 '/home/richardkxu/Downloads/tiny-imagenet-200/train/n02279972/images/n02279972_488.JPEG'
]

bottom10_labels = \
[
 'n02279972', 'n02279972', 'n02279972', 'n02279972', 'n02279972', 'n02279972',
 'n02279972', 'n02279972', 'n02279972', 'n02279972'
]


nrow = 5
ncol = 5

fig=plt.figure(figsize=(ncol*3, nrow*3))
base, img_name = os.path.split(test_img)
plt.axis('off')
plt.title(img_name)
fig.add_subplot(nrow, ncol, 1)
img = img_loader(test_img)
plt.axis('off')
plt.title('test_img')
plt.imshow(img)

for i in range(10):
    fig.add_subplot(nrow, ncol, i+6)
    img = img_loader(top10_imgs[i])
    plt.axis('off')
    plt.title(top10_labels[i])
    plt.imshow(img)

for i in range(10):
    fig.add_subplot(nrow, ncol, i+16)
    img = img_loader(bottom10_imgs[i])
    plt.axis('off')
    plt.title(bottom10_labels[i])
    plt.imshow(img)

plt.tight_layout()
plt.savefig('example4.png')
