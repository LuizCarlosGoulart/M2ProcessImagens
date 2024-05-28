import cv2
import numpy as np

def skin_color_thresholding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lower_bound = np.array([45, 52, 108], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)

    result = cv2.bitwise_and(img, img, mask=mask)

    return result, mask

def show_images(original, mask, result):
    cv2.imshow('Original Image', original)
    cv2.imshow('Skin Color Mask', mask)
    cv2.imshow('Detected Skin Regions', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_images(image_list):
    for image_path in image_list:
        img = cv2.imread(image_path)
        if img is not None:
            result, mask = skin_color_thresholding(img)
            output_filename = f"skin_threshold_{image_path}"
            cv2.imwrite(output_filename, result)
            print(f"Processed {image_path}, saved result as {output_filename}")
            #
            show_images(img, mask, result)
        else:
            print(f"Failed to load image {image_path}")

ImageList = ["face1.jpg", "face2.jpg", "face3.jpg"]

process_images(ImageList)
