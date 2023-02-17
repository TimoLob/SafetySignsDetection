import os
import shutil
output_images = "./data/output/images"
output_labels = "./data/output/labels"
annotations = "./data/annotated"


dirs = [output_images,output_labels,annotations]
#dirs = [annotations]


for d in dirs:
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.mkdir(d)


