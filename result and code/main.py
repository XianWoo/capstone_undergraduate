# ResNet152
import os
import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
# root_dir = '/home/seannang/code/emotion_recognition/dataset/basic/Image'
from getDataset import *
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.CenterCrop(180),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4681,0.4030,0.5275], std=[1.5016,1.5685,1.5956])])
train_set = RAF_Dataset(csv_file= '/mnt/emotion_recognition/dataset/basic/Image/train_set/train_label.csv',
                        root_dir='/mnt/emotion_recognition/dataset/basic/Image/train',
                        transform = transform)
# print(len(train_set))
# test_set = RAF_Dataset(csv_file= '/home/seannang/code/emotion_recognition/dataset/basic/Image/test_label.csv',
#                         root_dir='/home/seannang/code/emotion_recognition/dataset/basic/Image/test',
#                         transform = transform)
# print(len(test_set))
class_count = [1290,281,717,4772,1982,705,2524]
class_weight = 1./torch.tensor(class_count, dtype=torch.float)
# print(class_weight)
# print(train_set[0])
# sample = train_set[8][0].numpy().transpose(1,2,0)
# plt.imshow(sample)
# plt.show()

train_loaders = torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True)
images,labels=next(iter(train_loaders))
# print(labels)

surprise = fear = disgust = happiness = sadness = anger = neutral = 0
for images, labels in train_loaders:
    surprise += torch.sum(labels==0)
    fear += torch.sum(labels==1)
    disgust += torch.sum(labels==2)
    happiness += torch.sum(labels==3)
    sadness += torch.sum(labels==4)
    anger += torch.sum(labels==5)
    neutral += torch.sum(labels==6)
# print(surprise)
# grid = torchvision.utils.make_grid(images,nrows=1)
# sample = grid.numpy().transpose(1,2,0)
# plt.figure(figsize=(15,15))
# plt.imshow(sample)
# plt.show()

model = torchvision.models.resnet34(pretrained=True)
model.fc=nn.Sequential(nn.Linear(model.fc.in_features,500),nn.ReLU(),nn.Dropout(),nn.Linear(500,7))
# print(model)
optim = torch.optim.Adam(model.parameters(),lr=1e-2)

crit=nn.CrossEntropyLoss()
scalar = torch.cuda.amp.GradScaler()

num_epoches = 5
total_step = len(train_loaders)
losses=[]
train_acc=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epoches):
    for i,(images,labels) in enumerate(train_loaders):
        images = images.to(device)
        labels = labels.to(device)
        model = model.to(device)
        optim.zero_grad()
        with torch.cuda.amp.autocast():
            outputs=model(images)
            loss=crit(outputs,labels)
        loss.backward()
        optim.step()
        _,argmax=torch.max(outputs,1)
        acc=(labels==argmax.squeeze()).float().mean()
        losses.append(loss.item()/total_step)
        train_acc.append(acc.item())

        if(i+1)%37==1:
            print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f},Accuracy:{:.3f}'
                  .format(epoch+1,num_epoches,i+1,total_step,loss.item(),acc.item()))

plt.plot(losses)
plt.title("Loss Curve(batch_size=64,lr=0.0001)")
plt.xlabel("steps")
plt.ylabel("Loss")
plt.show()

plt.plot(train_acc)
plt.title("Training Curve(batch_size = 64,lr=0.0001)")
plt.xlabel("Steps")
plt.ylabel("Training Accuracy")
plt.show()



test_set = RAF_Dataset(csv_file= '/mnt/emotion_recognition/dataset/basic/Image/test_label.csv',
                        root_dir='/mnt/emotion_recognition/dataset/basic/Image/test',
                        transform = transform)
test_loaders=torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=False)

model.eval()
with torch.no_grad():
    correct=0
    total=0
    for images_test,labels_test in test_loaders:
        images_test = images_test.to(device)
        labels_test = labels_test.to(device)
        model = model.to(device)
        outputs_test = model(images_test)
        _,predicted = torch.max(outputs_test.data,1)
        total += labels_test.size(0)
        correct += (predicted == labels_test).sum().item()
    print('Acc of model is :{}%'.format(100*correct/total))
face_classes = 7
confusion_matrix = torch.zeros(face_classes,face_classes)
with torch.no_grad():
    for i, (images,labels) in enumerate(test_loaders):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs,1)
        for t,p in zip(labels.view(-1),preds.view(-1)):
            confusion_matrix[t.long(),p.long()] += 1
print(confusion_matrix)
print(confusion_matrix.diag()/confusion_matrix.sum(1))

plt.figure(figsize=(15,10))
class_names=['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']
df_cm = pd.DataFrame(confusion_matrix,index=class_names,columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm,annot=True,fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predict label')
torch.save(model.state_dict(),'resnet34_raf.pth')
        