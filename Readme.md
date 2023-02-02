# Safety Signs detection using machine learning methods

## Usage

1. Run "generator.py n" in pipeline 1, n is the number of signs that should be generated for each sign in Pipeline1\data\signs\hq
   1. n = 1000 => 1000 signs per class
2. Copy the content of the output folder in pipeline 1 into Pipeline2\data\Signs\Generated
3. Run "generatory.py n" in pipeline 2, n is the number of training images that should be generated
4. Create a Roboflow Project and upload the Pipeline2\data\output\ folder.
   1. Adjust data.yaml as needed with the class labels and the number of classes

5. Run the Train Yolov7 notebook
   1. adjust the roboflow project, project version and api key accordingly  
   2. I recommend creating a fork of this and the <https://github.com/TimoLob/yolov7> repository
   3. The weights will be downloaded at the end of the notebook
6. Run the Validate Yolov7 notebook
   1. By default the weights will be copies from a google drive location, adjust for your use
   2. This notebook will run the validation on a validation dataset and give you the option to upload your own files and annotate those


## Images

Images found in the thesis can be found in the images folder.