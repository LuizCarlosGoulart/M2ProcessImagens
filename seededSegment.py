import cv2
import numpy as np

def region_growing(image, seed_point, tolerance):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    seed_list = [seed_point]
    mask[seed_point] = 255
    
    mean_intensity = int(image[seed_point])
    
    while len(seed_list) > 0:
        x, y = seed_list.pop(0)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (0 <= x + dx < h) and (0 <= y + dy < w):
                    if mask[x + dx, y + dy] == 0:
                        intensity = int(image[x + dx, y + dy])
                        if abs(intensity - mean_intensity) <= tolerance:
                            mask[x + dx, y + dy] = 255
                            seed_list.append((x + dx, y + dy))
                            mean_intensity = (mean_intensity + intensity) // 2
    
    return mask

def seeded_segmentation(image_path, seed_point, tolerance=10):
    image = cv2.imread(image_path, 0)
    mask = region_growing(image, seed_point, tolerance)
    
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented_image

seed_point = (50, 50)
segmented_image = seeded_segmentation('person.jpg', seed_point)
cv2.imshow('Imagem segmentada', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
