'''a=1
while a<10:
    if a%2==0:
         if a*2%2==0:

            print("abc")
         print ("是偶数")
    elif a%2!=0:
      print ("是奇数")
    a=a+1
    '''
'''a, b = 0, 0
while b <= 100:
    a += b
    b = b + 1
print(a)
'''

'''
 guess, num = 1, 10
while guess != num:
    guess = int(input())
    if guess == num:
        print("猜对了")
    elif guess > num:
        print("猜大了")
    else:
        print("猜小了")
 '''
'''
num = 0
while num <= 10:  num += 1

else:
    print("大于十")
'''
'''
bands = ["NIKE", "adidas ", "xtep", "Ling"]
for band in bands:
  if band =="xtep":
     print ("在里面")
     break
  #print(band)

'''
'''
for x in range(10):
    print(x)
'''
'''
str="abcdefg"
it =iter(str)
for x in it:
    print(x,end=" ")
'''
''' 
import numpy as np

a1 = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])
print(a1)
print(a1 * 2)
print(a1 + a1)
'''

'''import numpy as np


def sig(x):
    return 1.0 / (1 + np.exp(-x))


flag = 1
while flag:
    inter = int(input())
    print(sig(inter))
'''
'''
import numpy as np
import cv2


img = cv2.imread("E:\\python\\wa.jpeg")   #读入图像。第二个参数cv2.IMREAD_COLOR/cv2.IMREAD_GRAYSCALE

cv2.imshow("image",img)   #显示图像。   注意：毫秒级的，如果没有下句则会闪退
cv2.waitKey(0)           #0--无限期的等待键盘输入，按键之后才会运行下一句
cv2.destroyAllWindows()   #删除任何我们建立的窗口。删除特定的窗口可以使用 cv2.destroyWindow()，在括号内输入你想删 除的窗口名。
'''
# coding=utf-8
'''
import torch
from torch.autograd import Variable

# requires_grad 表示是否对其求梯度，默认是True
x = Variable(torch.Tensor([3]), requires_grad=True)
y = Variable(torch.Tensor([5]), requires_grad=True)

z = 2 * x + y + 4

# 对x和y分别求导
z.backward()

# x的导数和y的导数
print("z对x的偏导:", x.grad.data)
print("z对y的偏导:", y.grad.data)
'''

#  from __future__ import print_function
# import torch

'''
import torch
import numpy as np

x = torch.empty(5, 3)
y = torch.rand(4, 6)
z = torch.zeros(4, 6, dtype=torch.float)
h = torch.tensor([5.5, 3])
q=x.new_ones(5,3, dtype=torch.double)
print(x, "\n", y, "\n", z, "\n", h,"\n",q)
print(x.size())
print(y+z)
print(torch.add(y,z))
print(y.__add__(z))
print(z[:,2]) #输出哪一行
#将张量转化为数组
a=torch.ones(3)
print(a)
b=a.numpy()
print(b)
a=a.__add__(4)
print(a)
print(b)
#将数组转化为张量
c=np.ones(4)
print(c)
d=torch.from_numpy(c)
np.add(c,1,out=c)
print(c)
print(d)
'''
# e=np.linspace
'''
import numpy as np
import cv2 as cv
img = cv.imread('plane1.jpg')
img2=cv.resize(img,(500,300),interpolation=cv.INTER_CUBIC)
gray= cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
akaze = cv.AKAZE_create()
kp, descriptor = akaze.detectAndCompute(gray, None)
img=cv.drawKeypoints(gray, kp, img2)
cv.imwrite('keypoints.jpg', img2)
'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('plane1.jpg')#飞机1
img1=cv.resize(img,(500,300),interpolation=cv.INTER_CUBIC)#放大
gray1= cv.cvtColor(img1, cv.COLOR_BGR2GRAY)#灰度化

img = cv.imread('plane2.jpg')#飞机2
img2=cv.resize(img,(500,300),interpolation=cv.INTER_CUBIC)#放大，与飞机1一样大
gray2= cv.cvtColor(img2, cv.COLOR_BGR2GRAY)#灰度化

#创建AKAZE特征检测器和描述符
akaze = cv.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(img1, None)
img3=cv.drawKeypoints(gray1, kp1, img1)#画出飞机1关键点并显示
cv.imwrite('keypoints1.jpg', img3)
kp2, des2 = akaze.detectAndCompute(img2, None)# BFMatcher with default params
img4=cv.drawKeypoints(gray2, kp2, img2)
cv.imwrite('keypoints2.jpg', img4)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)# Apply ratio
good_matches = []
for m,n in matches:
 if m.distance < 0.75*n.distance:
  good_matches.append([m])   # Draw matches
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('matches.jpg', img3)
# Select good matched keypoints
ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)# Compute homography
H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC,5.0)# Warp image
warped_image = cv.warpPerspective(img1, H, (img1.shape[1]+img2.shape[1], img1.shape[0]))
cv.imwrite('warped.jpg', warped_image)