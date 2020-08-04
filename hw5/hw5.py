import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace

workspace_dir = sys.argv[1]
image_dir = sys.argv[2]

train_mean = [0.34383293045079183, 0.45114563274311903, 0.555130165717435]
train_std = [0.2811158137405215, 0.27397806351580906, 0.27110181447699655]

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(15), #隨機旋轉圖片
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
    transforms.Normalize(mean=train_mean, std=train_std),
])
#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std),
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
        
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 2, 2]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'
        
        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform

    # 這個 FoodDataset 繼承了 pytorch 的 Dataset class
    # 而 __len__ 和 __getitem__ 是定義一個 pytorch dataset 時一定要 implement 的兩個 methods
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # 這個 method 並不是 pytorch dataset 必要，只是方便未來我們想要指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

# 給予 data 的路徑，回傳每一張圖片的「路徑」和「class」
def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels
train_paths, train_labels = get_paths_labels(os.path.join(workspace_dir, 'training'))

train_set = FoodDataset(train_paths, train_labels, mode='eval')
model = torch.load('./best.model')

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    # 最關鍵的一行 code
    # 因為我們要計算 loss 對 input image 的微分，原本 input x 只是一個 tensor，預設不需要 gradient
    # 這邊我們明確的告知 pytorch 這個 input x 需要gradient，這樣我們執行 backward 後 x.grad 才會有微分的值
    x.requires_grad_()
  
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

# 指定想要一起 visualize 的圖片 indices
img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

# 使用 matplotlib 畫出來
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([images, saliencies]):
    for column, img in enumerate(target):
        axs[row][column].imshow(img.permute(1, 2, 0).numpy())
plt.savefig(os.path.join(image_dir, "Q1.png"))
plt.show()
plt.close()

layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output

    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    model(x.cuda())
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
    x = x.cuda()
    x.requires_grad_()
    optimizer = Adam([x], lr=lr)
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
        objective = -layer_activations[:, filterid, :, :].sum()
        objective.backward()
        optimizer.step()
    filter_visualization = x.detach().cpu().squeeze()[0]

    hook_handle.remove()

    return filter_activations, filter_visualization

img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
filter_activations, filter_visualization = filter_explaination(images, model, cnnid=0, filterid=0, iteration=100, lr=0.1)

# 畫出 filter visualization
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(os.path.join(image_dir, "Q2-1.png"))
plt.show()
plt.close()
fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
for i, img in enumerate(filter_activations):
    axs[i].imshow(normalize(img))
plt.savefig(os.path.join(image_dir, "Q2-2.png"))
plt.show()
plt.close()

filter_activations, filter_visualization = filter_explaination(images, model, cnnid=5, filterid=0, iteration=100, lr=0.1)

# 畫出 filter visualization
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(os.path.join(image_dir, "Q2-3.png"))
plt.show()
plt.close()

# 畫出 filter activations
fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
for i, img in enumerate(filter_activations):
    axs[i].imshow(normalize(img))
plt.savefig(os.path.join(image_dir, "Q2-4.png"))
plt.show()
plt.close()

filter_activations, filter_visualization = filter_explaination(images, model, cnnid=10, filterid=0, iteration=100, lr=0.1)

# 畫出 filter visualization
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(os.path.join(image_dir, "Q2-5.png"))
plt.show()
plt.close()

# 畫出 filter activations
fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
for i, img in enumerate(filter_activations):
    axs[i].imshow(normalize(img))
plt.savefig(os.path.join(image_dir, "Q2-6.png"))
plt.show()
plt.close()

filter_activations, filter_visualization = filter_explaination(images, model, cnnid=10, filterid=10, iteration=100, lr=0.1)

# 畫出 filter visualization
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(os.path.join(image_dir, "Q2-7.png"))
plt.show()
plt.close()

# 畫出 filter activations
fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
for i, img in enumerate(filter_activations):
    axs[i].imshow(normalize(img))
plt.savefig(os.path.join(image_dir, "Q2-8.png"))
plt.show()
plt.close()

filter_activations, filter_visualization = filter_explaination(images, model, cnnid=10, filterid=30, iteration=100, lr=0.1)

# 畫出 filter visualization
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(os.path.join(image_dir, "Q2-9.png"))
plt.show()
plt.close()

# 畫出 filter activations
fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
for i, img in enumerate(filter_activations):
    axs[i].imshow(normalize(img))
plt.savefig(os.path.join(image_dir, "Q2-10.png"))
plt.show()
plt.close()

def predict(input):
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    return slic(input, n_segments=100, compactness=1, sigma=1)

img_indices = [2089, 2101, 8999, 9001]
images, labels = train_set.getbatch(img_indices)
fig, axs = plt.subplots(1, 4, figsize=(15, 8))
np.random.seed(16)

for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)

    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
    lime_img, mask = explaination.get_image_and_mask(
                                label=label.item(),
                                positive_only=False,
                                hide_rest=False,
                                num_features=11,
                                min_weight=0.05,
                            )
    axs[idx].imshow(lime_img)

plt.savefig(os.path.join(image_dir, "Q3.png"))
plt.show()
plt.close()

import torchvision
from torchvision import datasets, transforms
from torch import optim
from torch.nn import functional as F
from torchviz import make_dot
import shap

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

batch = next(iter(train_loader))
images, labels = batch
images.size()

background = images[:30]
e = shap.DeepExplainer(model, background.cuda())

iter_val_loader = iter(val_loader)
batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-1.png"))
print(label)

batch = next(iter_val_loader)
batch = next(iter_val_loader)
batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-11.png"))
print(label)

batch = next(iter_val_loader)
batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-2.png"))
print(label)

batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-3.png"))
print(label)

batch = next(iter_val_loader)
batch = next(iter_val_loader)
batch = next(iter_val_loader)
batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-4.png"))
print(label)

batch = next(iter_val_loader)
batch = next(iter_val_loader)
batch = next(iter_val_loader)
batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-5.png"))
print(label)

batch = next(iter_val_loader)
batch = next(iter_val_loader)
batch = next(iter_val_loader)
batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-6.png"))
print(label)

batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-7.png"))
print(label)

batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-8.png"))
print(label)

batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-9.png"))
print(label)

batch = next(iter_val_loader)
batch = next(iter_val_loader)
images, labels = batch
images.size()
test_images = images[0:1]
label = labels[0:1]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(os.path.join(image_dir, "Q4-10.png"))
print(label)
