# -*- coding: utf-8 -*-

# @Author: xyq

import os
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import torch.optim

train_dir = 'Train/'
test_dir = 'Test/'
train_label_file = 'TrainLabel.TXT'
test_label_file = 'TestLabel.txt'
brands = ['Audi','BMW','Benz']
types = ['Sedan','SUV']


class Mydataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.anno_files = annotations_file
        self.transform = transform

        self.size = 0
        self.name_list = []

        if not os.path.exists(self.anno_files):
            print(self.anno_files +  "does not exist!")
        with open(self.anno_files,'r') as f:
            for line in f:
                self.name_list.append(line)
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.root_dir + self.name_list[idx].split()[0]
        if not os.path.exists(img_path):
            print(img_path + 'not exist!')
            return None
        label_brand = int(self.name_list[idx].split()[1])
        label_type = int(self.name_list[idx].split()[2])
        image = Image.open(img_path)
        sample = {'image':image, 'brand':label_brand,'type':label_type}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,3,3)  # 498*498*3
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 249*249*3
        self.BN1 = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(3,6,3) # 247 * 247 * 6
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 123 * 123 *6
        self.BN2 = nn.BatchNorm2d(6)

        self.fc1 = nn.Linear(6*123*123,150)
        self.fc2 = nn.Linear(150,3)
        self.softmax1 = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(150,2)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.BN1(x)
        # x = self.relu1(x)
        # x = self.maxpool1(x)
        x = F.max_pool2d(F.relu(self.BN1(self.conv1(x))),2)
        # x = self.conv2(x)
        # x = self.BN2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        x = F.max_pool2d(F.relu(self.BN2(self.conv2(x))),2)
        x = x.view(-1,6*123*123) # 拉成一维
        # x = x.view(-1, self.num_flat_featuures(x))
        # x = self.fc1(x)
        # x = self.relu3(x)
        x = F.relu(self.fc1(x))
        x_brand = self.fc2(x)
        x_brand = self.softmax1(x_brand)
        x_type = self.fc3(x)
        x_type = self.softmax2(x_type)

        return x_brand,x_type

    def num_flat_featuures(self,x):
        size = x.size()[1:]
        nums_feature = 1
        for s in size:
            nums_feature *= s
        return nums_feature


def train(model,criterion,optimizer, schedule, num_epochs=50):
    loss_list = {'train':[],'test':[]}
    acc_brand = {'train':[],'test':[]}
    acc_type = {'train':[],'test':[]}
    best_model_w = copy.deepcopy(model.state_dict())
    best_acc = {"overall":0.0,'brand':0.0,'type':0.0}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print('-*' * 10)

        for phase in ['train','test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            correct_brand = 0
            correct_type = 0

            for data in data_loaders[phase]: # 遍历每个batch 16 16 16 12的图片
                inputs = data['image'].to(device)
                label_brand = data['brand'].to(device)
                label_type = data['type'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_brand, x_type = model(inputs)
                    x_brand = x_brand.view(-1,3)
                    x_type = x_type.view(-1,2)

                    _, preds_brand = torch.max(x_brand,1)
                    _, preds_type = torch.max(x_type,1)

                    loss = criterion(x_brand,label_brand) * 1.1 + 0.9 * criterion(x_type, label_type)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                correct_brand += torch.sum(preds_brand == label_brand)
                correct_type += torch.sum(preds_type == label_type)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            loss_list[phase].append(epoch_loss)

            epoch_acc_brand = correct_brand.item() / len(data_loaders[phase].dataset)
            epoch_acc_type = correct_type.item()/ len(data_loaders[phase].dataset)
            epoch_acc = 0.6 * epoch_acc_brand + 0.4 * epoch_acc_type

            acc_brand[phase].append(100 * epoch_acc_brand)
            acc_type[phase].append(100 * epoch_acc_type)
            print('{} Loss: {:.4f} Acc_brand: {:.2%}  Acc_type：{:.2%}'.format(phase, epoch_loss,epoch_acc_brand,epoch_acc_type))

            if phase == 'test' and epoch_acc > best_acc['overall']:
                best_acc["overall"] = epoch_acc
                best_acc['brand'] = epoch_acc_brand
                best_acc['type'] = epoch_acc_type
                best_model_w = copy.deepcopy(model.state_dict())
                print('Best val Acc:{:.2%} Best val brand Acc:{:.2%} Best val type Acc:{:.2%}'.
                      format(best_acc['overall'], best_acc['brand'],best_acc['type']))
    model.load_state_dict(best_model_w)
    torch.save(model.state_dict(),'best_model_w')
    print('Best val Acc:{:.2%} Best val brand Acc:{:.2%} Best val type Acc:{:.2%}'.
          format(best_acc['overall'], best_acc['brand'], best_acc['type']))
    return model, loss_list, acc_brand,acc_type


train_transforms = transforms.Compose([transforms.Resize((500,500)),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize((500,500)),transforms.ToTensor()])

transed_trainset = Mydataset(root_dir= train_dir, annotations_file=train_label_file,transform=train_transforms)
transed_testset = Mydataset(root_dir=test_dir,annotations_file=test_label_file,transform=test_transforms)
train_loader = DataLoader(dataset=transed_trainset,batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=transed_testset)
data_loaders = {'train':train_loader,'test': test_loader}
print("处理训练集数据：" + str(len(transed_trainset)))
print("处理测试集数据：" + str(len(transed_testset)))
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

network = Net().to(device)
optimizer = torch.optim.SGD(network.parameters(),lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
model, loss_list, acc_list_brand, acc_list_type = train(network, criterion, optimizer, exp_lr_scheduler, num_epochs=50)

