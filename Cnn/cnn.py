''' 挑战numpy100关'''
import PIL

'''import numpy as np #使用名称np导入numpy包
print(np.__version__)#打印出numpy版本号和配置信息
np.show_config()
#z=np.zeros(10)#创建一个空向量, 尺寸为10 (★☆☆)
#print(z)
z=np.zeros(10)
z[4]=1
z=np.arange(10,50)
z=z[::-1]
z=np.arange(9).reshape(3,3)
z=np.nonzero([0,1,2,3,4,5,0,8,9,0,12,0])
print(z)

#print("%d bytes"%(z.size*z.itemsize))
'''
'''#机器学习100天
import numpy as np
import pandas as pd
dataset = pd.read_csv('H:/DataSets/Data.csv')#//读取csv文件
x=dataset.iloc[:,:-1].values
print(x)
'''
''' python100天第2天'''
'''
import math
#f,r=map(int ,input('请输入两个数：').split())
f,r,year= (int(x) for x in input('请输入三个数:').split())
c=(f-32)/1.8
area=math.pi*r*r
perimeter=2*math.pi*r
is_leap=(year%4==0and year%100!=0 or year%400==0)
print('%.2f华氏度=%.2f摄氏度'%(f,c))
print('面积是%.2f,周长是%.2f'%(area,perimeter))
print(is_leap)
'''
import numpy as np
import torch
import cv2
'''
z=np.zeros(10)
z[4]=1
z=np.arange(50)
print(z)
z=z[::-1]
print(z)
z=np.arange(9).reshape(3,3)
print(z)
z=np.nonzero([1,2,0,0,4,0])
print(z)
z=np.ones(9).reshape(3,3)
print(z)
z=np.eye(3)
print(z)
z=np.random.random ((3,3,3))
print(z)
z=np.random.random((10,10))
print(z.min(),z.max())
z=np.ones((2,3,4,5),dtype=np.int)
print(z)
'''
'''
z=np.arange(10,30,5)
print(z)
z=np.random.random(30)
print(z.mean())
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
z=torch.rand(3,5)
print(z)
z=torch.empty(5,3)
print(z)
z=torch.zeros(2,4,dtype=torch.long)
print(z)
'''
'''
x=torch.ones(2,2,requires_grad=True)#可求导
print(x)
print(x.grad_fn)
y=-(x+2)
print(y)
y=y.mean()#标量
y.backward()#反向传播
print(x.grad)#张量关于x的导数

z=y*y*3
#out=z.mean()
#print(z,out)
#out.backward()
print(z)
z.backward()
print(x.grad)

a=torch.rand(2,2,requires_grad=True)
b=((a*3)/(a-1))
print(b)
print(b.grad_fn)
b=b.mean()
b.backward()
print(b.grad_fn)
print(a.grad)
'''
import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import functional as F
import cv2
import random

font = cv2.FONT_HERSHEY_SIMPLEX

root = '/public/yzy/coco/2014/train2014/'
annFile = '/public/yzy/coco/2014/annotations/instances_train2014.json'


# 定义 coco collate_fn
def collate_fn_coco(batch):
    return tuple(zip(*batch))


# 创建 coco dataset
coco_det = datasets.CocoDetection(root, annFile, transform=T.ToTensor())
# 创建 Coco sampler
sampler = torch.utils.data.RandomSampler(coco_det)
batch_sampler = torch.utils.data.BatchSampler(sampler, 8, drop_last=True)

# 创建 dataloader
data_loader = torch.utils.data.DataLoader(
    coco_det, batch_sampler=batch_sampler, num_workers=3,
    collate_fn=collate_fn_coco)

# 可视化
for imgs, labels in data_loader:
    for i in range(len(imgs)):
        bboxes = []
        ids = []
        img = imgs[i]
        labels_ = labels[i]
        for label in labels_:
            bboxes.append([label['bbox'][0],
                           label['bbox'][1],
                           label['bbox'][0] + label['bbox'][2],
                           label['bbox'][1] + label['bbox'][3]
                           ])
            ids.append(label['category_id'])

        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box, id_ in zip(bboxes, ids):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            cv2.putText(img, text=str(id_), org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                        thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
        cv2.imshow('test', img)
        cv2.waitKey()
