# by CEN Jun

import argparse
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='DEDUCE Scene_Only Evaluation')
parser.add_argument('--dataset',default='sun',help='dataset to test')
parser.add_argument('--hometype',default='home1',help='home type to test')
parser.add_argument('--floortype',default='data_0',help='data type to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')

global args
args = parser.parse_args()

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = 'resnet18_best_home.pth.tar'

model = models.__dict__[arch](num_classes=7)
checkpoint = torch.load(model_file)
print(checkpoint['best_prec1'])
print(checkpoint['epoch'])
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
#model.cuda()
model = torch.nn.DataParallel(model).cuda()
# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365_{}.txt'.format(args.envtype)

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

if(args.dataset == 'places'):
    data_dir = '/data/cenj/places365_train_2'
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
    correct,count=0,0
    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0)).cuda()
        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        if(classes[idx[0]] == class_name):
            correct+=1
        count+=1
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    correct_list.append(correct)
    totalnumber_list.append(count)
print('Average test accuracy is = {:2.2f}%'.format(100*sum(correct_list)/float(sum(totalnumber_list))))

