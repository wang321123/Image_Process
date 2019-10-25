'''
import cv2
from Stitcher import Stitcher
#import imutils
#if __name__ == "_main_":
# 读取拼接图片
imageA = cv2.imread('house1.jpg')
imageB = cv2.imread('house2.jpg')
#imageA = imutils.resize(imageA, width=400)
#imageB = imutils.resize(imageB, width=400)
# 把图片拼接成全景图
stitcher = Stitcher()
print(0)
(result,vis) = stitcher.stitch([imageA,imageB], showMatches=True)

#(result, vis) = Stitcher().stitch([imageA, imageB], showMatches=True)
# 显示所有图片
cv2.imwrite('image_A.jpg', imageA)
cv2.imwrite('Image B.jpg', imageB)
cv2.imshow("KeypointMatches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
import numpy as np
import imutils
import cv2


class Stitcher:
    #def __init__(self):
       # self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, image_a, image_b, ratio=0.75, reprojThresh=5.0,
               showMatches=False):

        # 简单地检测关键点并从两个图像中提取局部不变量描述符SIFT并匹配
        # (imageB, imageA) = images
        imageA = image_a
        imageB = image_b
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 没有足够的匹配关键点，返回空
        if M is None:
            #return None
            print(0)

        # 应用透视变换将图像缝合在一起
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检查是否应该可视化关键点匹配
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        # 返回缝合图像
        return result

    def detectAndDescribe(self, image):
        """
        :param image:
        :return: 特征描述点
        """
        # descriptor = cv2.xfeatures2d.SIFT_create()
        descriptor = cv2.ORB_create()
        # 检测SIFT特征点，并计算描述子
        # (kps, features) = descriptor.detectAndCompute(image, None)
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # matcher = cv2.DescriptorMatcher_create("BruteForce")
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 计算两组点之间的单应性需要  至少初始的四组匹配。
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        return None

    # 可视化
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 仅当关键点成功匹配时才处理匹配
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis


if __name__ == '__main__':
    imageA = cv2.imread('w2.png')

    img1 = cv2.resize( imageA , (500, 300), interpolation=cv2.INTER_CUBIC)
    gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)  # 灰度化

    imageB = cv2.imread('w1.png')

    img2 = cv2.resize(imageB, (500, 300), interpolation=cv2.INTER_CUBIC)
    gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    imageA = imutils.resize(imageA, width=400,height=500)
    imageB = imutils.resize(imageB, width=400,height=500)
    # 将图像缝合在一起以创建全景图
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch(imageA, imageB, showMatches=True)

    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
