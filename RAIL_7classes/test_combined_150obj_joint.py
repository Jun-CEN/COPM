# by CEN Jun
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import json
import datetime
import numpy as np
import csv

parser = argparse.ArgumentParser(description='150objects joint Combined Evaluation')
parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')

global args
args = parser.parse_args()

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = 'resnet50_best_150obj_joint_combined.pth.tar'

file_name='categories_places365_home.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

def load_dict(filename):
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic
best_prec1 = 0
one_hot=load_dict('150_7classes.json')

def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
#    print(feature.size())
    feature = feature.view(x.size(0), -1)
    output= model.fc(feature)
    return feature

class Object_Linear(nn.Module):
    def __init__(self):
        super(Object_Linear, self).__init__()
        self.fc1 = nn.Linear(22500, 8192)
        self.fc2 = nn.Linear(8192, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
#        out = self.dropout(out)
#        out = self.relu(out)
        out = self.fc2(out)
        return out
object_idt = Object_Linear()
#object_idt = torch.nn.DataParallel(object_idt).cuda()

class LinClassifier(nn.Module):
    def __init__(self,num_classes):
        super(LinClassifier, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, conv, idt):
        out = torch.cat((conv,idt),1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
classifier = LinClassifier(7)
#classifier = torch.nn.DataParallel(classifier).cuda()

model = models.__dict__[arch](num_classes=7)
checkpoint = torch.load(model_file)
model_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['model_state_dict'].items()}
obj_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['obj_state_dict'].items()}
classifier_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['classifier_state_dict'].items()}
model.load_state_dict(model_state_dict)
object_idt.load_state_dict(obj_state_dict)
classifier.load_state_dict(classifier_state_dict)
model.eval()
object_idt.eval()
classifier.eval()
model.cuda()
object_idt.cuda()
classifier.cuda()
print(checkpoint['best_prec1'])
# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if(args.dataset == 'places'):
    data_dir = '/data/cenj/places365_train'
    valdir = os.path.join(data_dir, 'val')
elif(args.dataset == 'sun'):
    data_dir = '/data/cenj/SUNRGBD'
    valdir = os.path.join(data_dir, 'test')
elif(args.dataset == 'vpc'):
    data_dir = vpc_dir
    home_dir = os.path.join(data_dir, 'data_'+args.hometype)
    valdir = os.path.join(home_dir,args.floortype)

correct_list = []
totalnumber_list = []
for class_name in os.listdir(valdir):
    correct,count,num_obj=0,0,0
    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0)).cuda()

        # forward pass
        output_conv = my_forward(model, input_img)
        row = one_hot[valdir+'/'+class_name+'/'+img_name]
        num_obj += sum(row)
        row = np.array(row)
        row = row.reshape(1,row.shape[0])
        column = row.T
        matrix = np.dot(column,row)
        obj_hot_vector=matrix.reshape(22500).tolist()
        t = torch.autograd.Variable(torch.FloatTensor(obj_hot_vector)).cuda()
        output_idt = object_idt(t)
        output_idt = output_idt.unsqueeze(0)
        logit = classifier(output_conv,output_idt)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        result=classes[idx[0]]
        if(result == class_name):
            correct+=1
        count+=1
    accuracy = 100*correct/float(count)
    avg_obj = num_obj/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
#    print('Average object number of {} class is {:2.2f}'.format(class_name,avg_obj))
    correct_list.append(correct)
    totalnumber_list.append(count)
print('Average test accuracy is = {:2.2f}%'.format(100*sum(correct_list)/float(sum(totalnumber_list))))

