import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from tqdm import tqdm
import threading
import math
import time
import threading
import sys

import random_perspective
import convert
import local_brightness

def adjust_brigthness(image,args):
    delta_brightness = 0.15
    if "delta_brightness" in args:
        delta_brightness = args["delta_brightness"]
    
    brightness_factor = random.uniform(1-delta_brightness,1+delta_brightness)

    bright_enhancer = ImageEnhance.Brightness(image)
    image = bright_enhancer.enhance(brightness_factor)
    return image

def adjust_contrast(image,args):
    delta_contrast = 0.15
    if "delta_contrast" in args:
        delta_contrast = args["delta_contrast"]
    contrast_factor = random.uniform(1-delta_contrast,1+delta_contrast)

    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(contrast_factor)
    return image


def apply_random_noise(image,args):

    mean = 0
    standard_deviation = 10
    if "mean" in args:
        mean = args["mean"]
    if "standard_deviation" in args:
        standard_deviation = args["standard_deviation"]
    image_data = np.array(image)

    # Add Gaussian noise with mean 0 and standard deviation 10 to the image
    noise = np.random.normal(mean, standard_deviation, image_data.shape)

    noisy_image = image_data + noise

    # Clip the noisy pixels to the valid range
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert the noisy image back to a PIL image
    noisy_image = Image.fromarray(noisy_image.astype("uint8"))

    return noisy_image

def random_rotation(image,args):
    expand = 1
    max_angle = 30
    if "expand" in args:
        expand = args["expand"]
    if "max_angle" in args:
        max_angle = args["max_angle"]
    
    angle = random.uniform(-1*max_angle, max_angle)
    image = image.rotate(angle,expand=expand)

    return image

def cutout(image,args):
    cutout_area = 0.2 # in % of image width/height
    cutout_colors = [(255, 255, 255, 0),(0, 0, 0)] # Transparent, Black
    if "cutout_area" in args:
        cutout_area = args["cutout_area"]
    if "cutout_colors" in args:
        cutout_colors = args["cutout_colors"]

    width,height = image.size
    max_cutout_width = int(cutout_area * width)
    max_cutout_height = int(cutout_area * height)

    cutout_width = random.randint(0,max_cutout_width)
    cutout_height = random.randint(0,max_cutout_height)

    cutout_x = random.randint(0,width-cutout_width)
    cutout_y = random.randint(0,height-cutout_height)

    img = ImageDraw.Draw(image)
    shape = [(cutout_x,cutout_y),(cutout_x+cutout_width,cutout_y+cutout_height)]

    color = random.choice(cutout_colors)
    img.rectangle(shape,fill=color)
    return image


def shear(image,args):
    max_x_shear = 0.5
    max_y_shear = 0.5

    if "max_x_shear" in args:
        max_x_shear = args["max_x_shear"]

    if "max_y_shear" in args:
        max_y_shear = args["max_y_shear"]

    matrix = [1,random.uniform(-max_x_shear,max_x_shear),0,random.uniform(-max_y_shear,max_y_shear),1,0]

    size = (image.size[0]*2,image.size[1]*2)
    image = image.transform(size,Image.Transform.AFFINE,matrix)
    return image


def random_perspective_transform(image,args):
    cv2image = convert.pil_to_opencv(image)
    transformed = random_perspective.random_perspective_transformation(cv2image)
    return convert.opencv_to_pil(transformed)

def add_light(image,args):
    cv2image = convert.pil_to_opencv(image)
    if random.random() < 0.5:
        transformed = local_brightness.add_parallel_light(cv2image)
    else:
        transformed = local_brightness.add_spot_light(cv2image)
    pil_image = convert.opencv_to_pil(transformed)
    
    return pil_image


def crop_to_content(image,args):
    image_box = image.getbbox()
    cropped = image.crop(image_box)
    return cropped

class Pipeline:
    def __init__(self,verbose=False):
        self.steps = []
        self.args = []
        self.odds = []
        self.verbose = True


    def add_step(self,function,odds,args):
        self.steps.append(function)
        self.args.append(args)
        self.odds.append(odds)

    def apply(self,image):
        for f,arg,odds in zip(self.steps,self.args,self.odds):
            if random.random() < odds:
                image = f(image,arg)
        return image


def create_pipeline():
    p = Pipeline()

    steps = [
        (adjust_brigthness,1,{"delta_brightness":0.8}),
        (adjust_contrast,1,{"delta_contrast":0.8}),
        (apply_random_noise,0.9,{}),
        (add_light,0.5,{}),
        (cutout,0.7,{"cutout_area":0.4}),
        (random_rotation,0.95,{}),
        (random_perspective_transform,0.95,{}),
        (crop_to_content,1,{}),
    ]

    for step in steps:
        p.add_step(*step)
    return p




def generate_images(number_of_images,path_to_base_image,pipeline,output_folder):
    base,ext = os.path.splitext(os.path.basename(path_to_base_image))

    image = Image.open(path_to_base_image)
    image = image.convert("RGBA")
    for i in tqdm(range(number_of_images),desc=base):
        
        modified_image = pipeline.apply(image.copy())
        out_path = os.path.join(output_folder,str(i)+"_"+base+".png")
        modified_image.save(out_path)


if __name__ == "__main__":
    sign_dir = "./data/Signs/hq/" 
    output_dir = "./data/output/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    N = 500

    if len(sys.argv) > 1:
        N = int(sys.argv[1])

    sign_files = os.listdir(sign_dir)
    p = create_pipeline()
    threads = []
    for sign in sign_files:
        base,ext = os.path.splitext(sign)
        base_image_path = os.path.join(sign_dir,sign)
        out_dir = os.path.join(output_dir,base)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        thread = threading.Thread(target=generate_images,args=(N,base_image_path,p,out_dir))
        thread.start()
    for i in range(len(threads)):
        threads[i].join()

        




