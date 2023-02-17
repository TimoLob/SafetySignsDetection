import cv2
import numpy as np
from PIL import Image

"""
Helper functions that convert between PIL and cv2 images
"""


def get_transparency(image):
    img = np.array(image)  
    return img[:,:,3].copy()



def pil_to_opencv(image):
    img = np.array(image)    # 'img' Color order: RGBA
    red = img[:,:,0].copy()        # Copy R from RGBA
    img[:,:,0] = img[:,:,2].copy() # Copy B to first order. Color order: BGBA
    img[:,:,2] = red               # Copy R to the third order. Color order: BGRA
    opencv_image = img              # img is OpenCV variable
    return opencv_image
    


def opencv_to_pil(image):
    img = np.array(image)
    red = img[:,:,2].copy()   
    img[:,:,2] = img[:,:,0].copy()
    img[:,:,0] = red

    pil_image = Image.fromarray(img)
    return pil_image
    

if __name__ == "__main__":
    input_path = './data/signs/hq/0.png'
    image = Image.open(input_path)
    image = image.convert("RGBA")

    cv2image = pil_to_opencv(image)

    pilimage = opencv_to_pil(cv2image)
