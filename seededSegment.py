<<<<<<< HEAD
import numpy as np
import cv2
from tqdm import tqdm
from queue import Queue


def checkPoint(img, point, visited):
    if 0 <= point[0] < img.shape[0] and 0 <= point[1] < img.shape[1]:
        if visited[point[0], point[1]] == 1:
            return 0
        return 1
    return 0


def updatePixel(img, point, tolerance, result, average, count):
    B = abs(int(img[point[0], point[1], 0]) - average[0] / count)
    G = abs(int(img[point[0], point[1], 1]) - average[1] / count)
    R = abs(int(img[point[0], point[1], 2]) - average[2] / count)

    if B < tolerance and G < tolerance and R < tolerance:
        result[point[0], point[1], 0] = img[point[0], point[1], 0]
        result[point[0], point[1], 1] = img[point[0], point[1], 1]
        result[point[0], point[1], 2] = img[point[0], point[1], 2]
        average[0] += float(img[point[0], point[1], 0])
        average[1] += float(img[point[0], point[1], 1])
        average[2] += float(img[point[0], point[1], 2])
        count += 1
        return 1, result, average, count
    return 0, result, average, count


def ShowResults(filename, result):
    cv2.imwrite(filename, result)


def SeedPointSegmentation(img, seedPoint, tolerance, imgNameOut):
    Tolerance = tolerance
    Visited = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    Result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    Average = [float(img[seedPoint[0], seedPoint[1], 0]), float(img[seedPoint[0], seedPoint[1], 1]), float(img[seedPoint[0], seedPoint[1], 2])]
    Count = 1

    Q = Queue()
    Q.put([seedPoint[0], seedPoint[1]])
    Visited[seedPoint[0], seedPoint[1]] = 1

    while Q.qsize() > 0:
        CurrentPoint = Q.get()
        Ret, Result, Average, Count = updatePixel(img, CurrentPoint, Tolerance, Result, Average, Count)
        if Ret == 0:
            continue

        p1 = [CurrentPoint[0] + 1, CurrentPoint[1]]
        p2 = [CurrentPoint[0] - 1, CurrentPoint[1]]
        p3 = [CurrentPoint[0], CurrentPoint[1] - 1]
        p4 = [CurrentPoint[0], CurrentPoint[1] + 1]

        if checkPoint(img, p1, Visited):
            Q.put(p1)
            Visited[p1[0], p1[1]] = 1
        if checkPoint(img, p2, Visited):
            Q.put(p2)
            Visited[p2[0], p2[1]] = 1
        if checkPoint(img, p3, Visited):
            Q.put(p3)
            Visited[p3[0], p3[1]] = 1
        if checkPoint(img, p4, Visited):
            Q.put(p4)
            Visited[p4[0], p4[1]] = 1

    ShowResults(imgNameOut, Result)


# Lista de imagens
ImageList = ["face1.jpg", "face2.jpg", "face3.jpg"]

# Configurações de teste
test_configs = [
    {"image": ImageList[0], "seedPoint": [600, 950], "tolerance": 120},
    {"image": ImageList[1], "seedPoint": [250, 250], "tolerance": 100},
    {"image": ImageList[2], "seedPoint": [400, 350], "tolerance": 100},
]

for i, config in enumerate(test_configs):
    Image = cv2.imread(config["image"])
    if Image is not None:
        print(f"Processing {config['image']} with seed point {config['seedPoint']} and tolerance {config['tolerance']}")
        output_filename = f"out_seed_seg_{i+1}.png"
        SeedPointSegmentation(Image, config["seedPoint"], config["tolerance"], output_filename)
    else:
        print(f"Failed to load image {config['image']}")
=======
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
>>>>>>> c6df975d026de2fc52c799f5041aa9c34f092c0a
