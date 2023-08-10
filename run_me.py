from Detector import *
#modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz"
modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz"
classfile = "coco.names"
#imagepath = "test/image1.jpg"
videopath = 0# for webcam
#videopath="test/"
threshold = 0.5 
detector=Detector()
detector.readClasses(classfile)
detector.dowmloadmodel(modelURL)
detector.loadModel()
#detector.predictImage(imagepath,threshold)
detector.predictVideo(videopath,threshold)


'''import tensorflow as tf

print("Num GPUs Available:", len(tfw.config.list_physical_devices('GPU')))
'''

