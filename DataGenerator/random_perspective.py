import cv2
import numpy as np
import random


def perspective_transform(img, src_points, dst_points):
    height, width = img.shape[:2]
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def crop_image(img, min_x, min_y, new_width, new_height):
    min_x, min_y = int(min_x), int(min_y)
    new_width, new_height = int(new_width), int(new_height) 
    return img[min_y:min_y+new_height, min_x:min_x+new_width]

def save_image(img, path):
    cv2.imwrite(path, img)


def random_perspective_transformation(img):
    height, width = img.shape[:2]
    # Define the source and destination points for the perspective transformation
    # The 4 corners are shifted to a random point inside their respective quadrant
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_points = np.float32([[random.randint(0, width//4), random.randint(0, height//4)],
                            [width - random.randint(0, width//4), random.randint(0, height//4)],
                            [width - random.randint(0, width//4), height - random.randint(0, height//4)],
                            [random.randint(0, width//4), height - random.randint(0, height//4)]])

    # Perform the perspective transformation
    result = perspective_transform(img, src_points, dst_points)
    # Calculate new image size
    min_x, min_y = dst_points[0][0], dst_points[0][1]
    max_x, max_y = dst_points[0][0], dst_points[0][1]
    for point in dst_points:
        min_x, min_y = min(min_x, point[0]), min(min_y, point[1])
        max_x, max_y = max(max_x, point[0]), max(max_y, point[1])
    new_width, new_height = max_x - min_x, max_y - min_y

    # Crop image 
    result = crop_image(result, min_x, min_y, new_width, new_height)
    
    return result

if __name__ == "__main__":
    input_path = './data/signs/hq/0.png'
    output_path = './data/output/0.png'

    img = cv2.imread(image_path,-1)
    result = random_perspective_transformation(img)
    save_image(result, output_path)