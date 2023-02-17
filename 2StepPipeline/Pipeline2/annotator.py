import yaml
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


class AnnotationProperties:
    def __init__(self, path_to_yaml:str):
        self.path_to_yaml = path_to_yaml
        with open(path_to_yaml) as f:
            yaml_data = yaml.safe_load(f)
            self.labels = yaml_data['names']
            self.num_classes = len(self.labels)
        
        # Choose different colors for each class
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)]
        self.colors = self.colors[:self.num_classes]

    def get_label(self, label_id:int):
        return self.labels[label_id]

    def get_color(self, label_id:int):
        return self.colors[label_id]
    
    def get_num_classes(self):
        return self.num_classes

    def get_label_id(self, label:str):
        return self.labels.index(label)
    
    def get_labels(self):
        return self.labels


class ImageAnnotator:
    def __init__(self,ann_properties):
        self.annotations = ann_properties
        self.font = ImageFont.truetype("arial.ttf", 25)

    def annotate_image(self,path_to_image:str, path_to_label:str):
        """Annotate image with label.

        Args:
            path_to_image: Path to image.
            path_to_label: Path to label.
            label_id_to_string: Dictionary mapping label id to label string.

        Returns:
            Annotated image.
        """
        image = Image.open(path_to_image)
        draw = ImageDraw.Draw(image)

        if not os.path.exists(path_to_label):
            return image
        
        with open(path_to_label) as f:
            for line in f:
                label_id, x, y, w, h = line.split() # x,y,w,h and in percent of the image size
                label_id, x, y, w, h = int(label_id), float(x), float(y), float(w), float(h)
                #print(x,y,w,h)
                x1, y1 = (x - w / 2) * image.width, (y - h / 2) * image.height
                x2, y2 = (x + w / 2) * image.width, (y + h / 2) * image.height

                label:str = self.annotations.get_label(label_id)
                color = self.annotations.get_color(label_id)

                draw.rectangle(((x1, y1), (x2, y2)), outline=color)
                #print(x1,y1,x2,y2)
                
                draw.text((x1, max(y1 - 25,0)), label, fill=color, font=self.font)
                

        return image


if __name__ == "__main__":

    image_folder = "./data/output/Images/"
    label_folder = "./data/output/Labels/"
    yaml_file = "./data/output/data.yaml"
    output_folder = "./data/annotated"

    images = os.listdir(image_folder)
    labels = os.listdir(label_folder)
    annotation_properties = AnnotationProperties(yaml_file)

    annotator = ImageAnnotator(annotation_properties)
    for image_file in tqdm(images):
        label = os.path.splitext(image_file)[0]+".txt"
        image = annotator.annotate_image(image_folder + image_file, label_folder + label)
        #image.show()
        image.save(os.path.join(output_folder,image_file))
        
        
        
