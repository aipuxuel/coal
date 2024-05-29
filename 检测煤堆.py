import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_depth_image(original_image_path, depth_image_path, lower_threshold, upper_threshold):
    # 读取原图和深度图
    original_image = cv2.imread(original_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    # 确保深度图是单通道图像
    if len(depth_image.shape) == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    # 创建一个掩膜，像素值在(lower_threshold, upper_threshold]之间的设为0，其他设为255
    mask = cv2.inRange(depth_image, lower_threshold + 1, upper_threshold)
    # 将满足条件的像素点设为黑色，其他设为白色
    processed_depth_image = np.where(mask > 0, 0, 255).astype(np.uint8)
    # 找到黑色区域的轮廓
    contours, _ = cv2.findContours(processed_depth_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个和原图大小相同的空白图像
    contour_image = np.zeros_like(original_image)
    # 在空白图像上绘制绿色轮廓
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    # 将轮廓加到原图上
    result_image = cv2.addWeighted(original_image, 1, contour_image, 1, 0)
    # 计算黑色像素值的个数
    num_black_pixels = np.sum(processed_depth_image == 0)
    # 如果黑色像素值个数大于1000，打印提示信息
    if num_black_pixels > 10000:
        print("有煤矿堆积！")
    else:
        print("没有煤矿堆积。")
    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image with Contours')
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Processed Depth Image')
    plt.imshow(processed_depth_image, cmap='gray')
    plt.axis('off')
    plt.show()
    # 返回处理后的深度图像和带有轮廓的原图
    return processed_depth_image, result_image

# 示例用法
original_image_path = 'img_274.jpg'
depth_image_path = 'rawdepth_274.png'
lower_threshold = 115
upper_threshold = 150

processed_depth_image, original_with_contours = process_depth_image(original_image_path, depth_image_path, lower_threshold, upper_threshold)

# 可选：保存处理后的深度图像和带有轮廓的原图
cv2.imwrite('processed_depth_image.png', processed_depth_image)
cv2.imwrite('original_with_contours.png', original_with_contours)
