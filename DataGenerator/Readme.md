# Data Generator

## Usage

1. Update data/output/data.yaml with the class labels in order and the number of classes
2. Put input sign images in data/signs
   1. The first character of the filename MUST be the class index according to the order in data.yaml
3. Run python generator.py #Images #classes
   1. Default values if not provided : 1000 7
4. The output data in yolov5 format can be found in data/output
   

annotator.py can annotate the images in output for visualization.  
clear_output.py deletes everything in data/output and data/annotated.


## Training Data on Roboflow

https://universe.roboflow.com/signs-ipufk/synth-rcgnr


## Validation Data (TGA)

https://universe.roboflow.com/sicherheitskennzeichnung/safety-signs-germany