import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
# 1为彩色通道
img=cv2.imread('E:/photo/crow.jpg',1)
cv2.imshow('crow',img)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()
# 0为灰度图
img=cv2.imread('E:/photo/crow.jpg',0)
cv2.imshow('crow',img)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()
# 使用matplotlib打印操作
plt.imshow(img)
# 不同通道的操作
img=cv2.imread('E:/photo/crow.jpg',1)
B,G,R=cv2.split(img)
img_rgb=cv2.merge((R,G,B))
plt.imshow(img_rgb)
img=cv2.imread('E:/photo/crow.jpg',1)
B,G,R=cv2.split(img)
img_new_gbr=cv2.merge((G,R,B))
plt.imshow(img_new_gbr)
# 图像的存储方式
print(img)
print(img.dtype)
print(img.shape)
# 图像的裁剪 image crop
img_crop=img[200:400,200:400]
cv2.imshow('img_crop',img_crop)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()
# 统计直方图
img_gray=cv2.imread('E:/photo/crow.jpg',0)
hist=img_gray.flatten()
plt.hist(hist,256,[0,256])
# 统计直方图平均化
img_eq=cv2.equalizeHist(img_gray)
cv2.imshow('hist_eq',img_eq)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()
hist=img_eq.flatten()
plt.hist(hist,256,[0,256])
# 对R通道的变化
img=cv2.imread('E:/photo/crow.jpg',1)
B,G,R=cv2.split(img)
const=155
R[R>100]=255
R[R<=100]=R[R<=100]+155#调亮图片
img_new_R=cv2.merge((R,G,B))
plt.imshow(img_new_R)
# gamma correction 伽马矫正

img_dark = cv2.imread('E:/photo/crow.jpg')
cv2.imshow('img_dark', img_dark)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)#将gamma矫正后的像素点存储进入table中
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)
img_brighter = adjust_gamma(img_dark, 2)
cv2.imshow('img_brighter', img_brighter)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
# 相似 仿射 投影变换
img= cv2.imread('E:/photo/crow.jpg')

M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

print(M)

print(M)
img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna2', img_rotate2)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5)
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

print(M)

rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affine lenna', dst)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


def random_warp(img, row, col):
    height, width, channels = img.shape

    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp


M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()