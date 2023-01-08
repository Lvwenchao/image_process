import cv2
import numpy as np


def ums(image_path, w=0.6):
    src = cv2.imread(image_path)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    # cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    cv2.imshow("mask image", usm)

    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0:h, 0:w, :] = src
    result[0:h, w:2 * w, :] = usm
    # cv2.putText(result, "original image", (10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    # cv2.putText(result, "sharpen image", (w + 10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    # cv.putText(图像名，标题，（x坐标，y坐标），字体，字的大小，颜色，字的粗细）

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("../sample_data/filter_re.png", usm)


ums("../sample_data/filter_test.png")
