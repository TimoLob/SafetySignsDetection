import os
import random
import sys
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import threading
import math
import datetime

import perlin_noise

class Label:
    def __init__(self,id,x,y,w,h):
        self.id = id
        # x,y,w,h in % of image size
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class ImageLabel:
    def __init__(self,image_dim,path=None):
        self.labels = []
        if path:
            with open(path) as f:
                for line in f:
                    label_id, x, y, w, h = line.split()
                    label_id, x, y, w, h = int(label_id), float(x), float(y), float(w), float(h)
                    
                    self.labels.append(Label(label_id,x,y,w,h,image_dim))


    def add_label(self,label):
        self.labels.append(label)

    def __repr__(self):
        s = ""
        for l in self.labels:
                s+=f"{l.id} {l.x} {l.y} {l.w} {l.h}\n"
        return s

    def write(self,path):
        if len(self.labels)>0:
            with open(path,"w") as f:
                for l in self.labels:
                    f.write(f"{l.id} {l.x} {l.y} {l.w} {l.h}\n")





class SignFactory():
    def __init__(self,sign_dir,number_classes):
        self.sign_dir = sign_dir
        self.nc = number_classes

        self.sign_sub_dirs = [name for name in os.listdir(sign_dir) if os.path.isdir(os.path.join(sign_dir, name))]
        print("Label - Number of Images")
        for i in range(len(self.sign_sub_dirs)):
            print(i,"-",len(os.listdir(os.path.join(self.sign_dir,self.sign_sub_dirs[i]))))

    def get_random_sign(self,c=None):
        if not c:
            c = random.choice(self.sign_sub_dirs)
        else:
            c = str(c)
        
        images = os.listdir(os.path.join(self.sign_dir,c))
        return int(c),os.path.join(self.sign_dir,c,random.choice(images))


def enhance_image(image):
    # Randomly adjust brightness and saturation of image

    delta_brightness = 0.15
    brightness_factor = random.uniform(1-delta_brightness,1+delta_brightness)

    delta_contrast = 0.15
    contrast_factor = random.uniform(1-delta_contrast,1+delta_contrast)

    bright_enhancer = ImageEnhance.Brightness(image)
    image = bright_enhancer.enhance(brightness_factor)

    bright_enhancer = ImageEnhance.Contrast(image)
    image = bright_enhancer.enhance(contrast_factor)

    return image



def add_random_noise(image):
    # Convert the image to a numpy array
    image_data = np.array(image)

    # Add Gaussian noise with mean 0 and standard deviation 10 to the image
    noise = np.random.normal(0, 3, image_data.shape)
    noisy_image = image_data + noise

    # Clip the noisy pixels to the valid range
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert the noisy image back to a PIL image
    noisy_image = Image.fromarray(noisy_image.astype("uint8"))

    return noisy_image

def apply_random_distortion(image):
    # Flip the image horizontally with probability 0.5
    #if random.random() > 0.5:
    #    image = image.transpose(Image.FLIP_LEFT_RIGHT)


    # Adjust Brightness and Contrast
    image = enhance_image(image)

    # Rotate the image by a random angle between -15 and 15 degrees
    # angle = random.uniform(-30, 30)
    # image = image.rotate(angle,expand=1)

    # Add random noise to the image
    #image = add_random_noise(image)
    #noise_mulitplyer = 10
    #image_data = np.array(image)
    #noise = perlin_noise.perlin_noise_grid(*image.size)
    #noisy_image = image_data + noise_mulitplyer * noise
    #noisy_image = np.clip(noisy_image, 0, 255)
    #noisy_image = Image.fromarray(noisy_image.astype("uint8"))

    return image


def distort_background(image):
    if random.random() > 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = enhance_image(image)
    angle = random.uniform(-15, 15)
    image = image.rotate(angle,expand=0)
    if random.random()<0.1:
        image = add_random_noise(image)
    return image

def overlay_image(sign_image,background_image,position,sign_size):
    x1,y1 = position

    sign_image=sign_image.resize(sign_size)

    background_image.alpha_composite(sign_image,dest=(x1,y1))

    return background_image

def intersects(box1,box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    x_overlap = (x1 <= x2 + w2) and (x2 <= x1 + w1)
    y_overlap = (y1 <= y2 + h2) and (y2 <= y1 + h1)
    return x_overlap and y_overlap


def generate_image(filename,sign_factory):
    global background_files
    if(len(background_files)==0):
        print("Reloading... repeating backgrounds")
        background_files = os.listdir(backgrounds_dir)

    # Choose a random background image
    background_file = random.choice(background_files)
    background_files.remove(background_file)
    background_image = Image.open(os.path.join(backgrounds_dir, background_file))
    #print(os.path.join(backgrounds_dir, background_file))
    background_image = background_image.convert("RGBA")
    background_image = background_image.resize((640,640),resample=Image.Resampling.BICUBIC)

    # Create a Label
    bg_width,bg_height = background_image.size
    label = ImageLabel(background_image.size)

    number_of_objects_of_interest = random.choice(number_of_objects_of_interest_probabilities)


    image = distort_background(background_image)

    bboxes = []
    for _ in range(number_of_objects_of_interest):
        label_id, sign_path = sign_factory.get_random_sign()
        sign_image = Image.open(sign_path)
        sign_image = sign_image.convert("RGBA")

        sign_width,sign_height = sign_image.size
        aspect_ratio = sign_width/sign_height

        # Randomly select a position and scale for the image
        scale = random.randint(25,100) # in Absolute pixels

        size = (int(scale*random.uniform(0.9,1.1)),int(scale*random.uniform(0.9,1.1/aspect_ratio)))
        
        pos = (int(random.uniform(0,bg_width-size[0])),int(random.uniform(0,bg_height-size[1])))

        w,h = (size[0]/bg_width), (size[1]/bg_height)
        x,y =  (pos[0]/bg_width)+w/2, (pos[1]/bg_height)+h/2 #center
        overlap = False
        for box in bboxes:
            if intersects(box,(pos[0],pos[1],w,h)):
                overlap=True
                break
        if overlap:
            continue
        bboxes.append((pos[0],pos[1],w,h))
        label.add_label(Label(label_id,x,y,w,h))
        
        # Overlay the distorted sign image on the background image
        image = overlay_image(sign_image, background_image,pos,size)

    
    # Save the composite image
    image.save(os.path.join(output_image_dir, filename+".png"))
    label.write(os.path.join(output_label_dir, filename+".txt"))



# Directory containing background images
backgrounds_dir = "./data/Backgrounds/" #"./data/firenet_backgrounds/images/"#"./data/Stanford Background Dataset/images"

# List of background files
background_files = os.listdir(backgrounds_dir)

number_of_objects_of_interest_probabilities = [0]*5 + [1]*25+ [2]*50+ [3]*5 # 5% no signs, 65% 1 sign, 25% 2 signs, 5% 3 signs

prefix = "_"+str(int(random.random()*1000))

# output_dir
output_dir = "./data/output/"
output_image_dir = os.path.join(output_dir,"images/")
output_label_dir = os.path.join(output_dir,"labels/")

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

def generate_images(number_of_images,signfactory,start=0):
    for i in tqdm(range(start,start+number_of_images)):
            generate_image(str(i)+prefix,signfactory)

if __name__ == "__main__":

    N = 1000
    if len(sys.argv) > 1:
        N = int(sys.argv[1])

    sf = SignFactory("./data/Signs/Generated",7)
    print("#Backgrounds:",len(os.listdir(backgrounds_dir)))

    num_threads = 4
    batch_size = math.ceil(N/num_threads)
    threats = []
    for i in range(num_threads):
        start = i*batch_size
        num_images = min(batch_size,N-start)
        print(start,num_images)

        thread = threading.Thread(target=generate_images,args=(num_images,sf,start))
        threats.append(thread)
        thread.start()

    for i in range(num_threads):
        threats[i].join()
            

    