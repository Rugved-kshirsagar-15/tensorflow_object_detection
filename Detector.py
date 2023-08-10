import cv2, time,os , tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __int__(self):
        pass

    def readClasses(self,classesfilepath):
        with open(classesfilepath,'r') as f:
            self.classesList = f.read().splitlines()

         # Colors list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))   

        #print(len(self.classesList),len(self.colorList))

    def dowmloadmodel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        #print(fileName)
        #print(self.modelName)

        self.cacheDir = "./pretrained_models"
        
        os.makedirs(self.cacheDir ,exist_ok=True)

        get_file(fname=fileName, 
        origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoits", extract=True)

    def loadModel(self):
        print("Loading model" + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoits", self.modelName, "saved_model"))

        print("model"+ self.modelName + "loaded successfully....")
    
    # create boundigbox

    def createBoundigBox(self, image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=40,iou_threshold=threshold,score_threshold=threshold)

        print(bboxIdx)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLablelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLablelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax *imW, ymin * imH ,ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax),int(ymin), int(ymax)

                cv2.rectangle(image,(xmin,ymin),(xmax,ymax), color=classColor, thickness=1,)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor,2)

        return image        


                #print(ymin, xmin, ymax, xmax)
                #break

     #predict img
    '''def predictImage(self ,imagepath,threshold = 0.5):
        image = cv2.imread(imagepath)

        bboxImage = self.createBoundigBox(image,threshold)

        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''


   # predict video
    def predictVideo(self, videopath, threshold = 0.5):
        cap = cv2.VideoCapture(videopath) #read a frame
         # check frame is working or not 
        if (cap.isOpened() == False):
            print("Error opening  file...")
            return
        (success, image) = cap.read()
        # set the video frame counter to zero
        startTime = 0 

        while success:
            currentTime = time.time()
            #calculate the number of frames per second
            fps= 1/(currentTime- startTime)
            startTime = currentTime

            bboxImage = self.createBoundigBox(image, threshold)
             #write the calculated number of frames per second of frame
            cv2.putText(bboxImage, "FPS:" + str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            cv2.imshow("result", bboxImage) #disply the frame
            # wait for 1ms .if a key is pressed, retreive the ASCII code of the key. 
            key = cv2.waitKey(1)& 0xFF
            if key == ord("w"): # 'w' button is pressed to close the frame
                break # break the loop
            (success, image) = cap.read()#release  the videocapture object
        cv2.destroyAllWindows()  # close the windows






