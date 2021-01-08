import ros2bag as rosbag
import yaml
import cv2 as cv
from cv_bridge import CvBridge
import imutils
import numpy as np
import matplotlib.pyplot as plt
from object_vision_sensing.camera_reconstruct.cameraReconstruct import cameraReconstructor


class objectDetector(object):
    def __init__(self, maxNumObjects = 14):
        self.maxNumObjects = maxNumObjects
        self.bridge = CvBridge()
        self.reconstructor = cameraReconstructor()
        self.croppedRect = None
    
    # set the region of interest for cropping
    def setCroppedRect(self, croppedRect):
        self.croppedRect = croppedRect
        
    def setReconstructor(self, cameraIntrinsics = None, rotation = None, transition = None):
        self.reconstructor.reset(cameraIntrinsics=cameraIntrinsics, rotation=rotation, transition=transition)
    
    # extract image from rosbag
    def imageFromBag(self, bagPath, channel, frameNum = 0):
        bag = rosbag.Bag(bagPath, 'r')
        count = 0
        messages = bag.read_messages(topics=['/'+channel+'/image_raw'])
        for topic, msg, t in messages:
            if count >= frameNum:
                break
            count = count + 1
        if msg.encoding == 'bgra8':
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        else:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        cv_img = self.depthConvert(cv_img, msg.encoding)
        
        return cv_img
    
    def cameraInfoFromBag(self, bagPath, channel, frameNum = 0):
        bag = rosbag.Bag(bagPath, 'r')
        count = 0
        messages = bag.read_messages(topics=['/'+channel+'/camera_info'])
        for topic, msg, t in messages:
            if count >= frameNum:
                break
            count = count + 1
        cameraIntrinsics = np.array(msg.K).reshape(3,3)
        return cameraIntrinsics
        
    def plotColorImage(self, colorimg):
        plt.imshow(colorimg)
    
    def cropImage(self, inputImg, croppedRect):
        assert type(croppedRect) == np.ndarray
        [x,y,dx,dy] = croppedRect.astype('int')
        img = inputImg[y:y+dy,x:x+dx]
        return img

    # detect the centroid of the objects in the image, output are the coordinates of the centroids, and a list of opencv contour objects
    def centroidDetection(self, image, maxNumObjects):
        image = image.copy()
        image = image.astype('uint8')
        blurred = cv.GaussianBlur(image, (5, 5), 0)
        thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)[1]
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [cnt for cnt in cnts if cv.contourArea(cnt)>5.0]
        cnts = sorted(cnts, key = lambda x: -cv.contourArea(x))
        filteredCount = min(len(cnts), maxNumObjects)
        cnts = cnts[:filteredCount]
        centroids = np.zeros((filteredCount, 2), dtype = int)
        for cntId in range(len(cnts)):
            cnt = cnts[cntId]
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids[cntId,0] = cx
            centroids[cntId,1] = cy
        return centroids, cnts       
    
    def detectionPlot(self, image, centroids, plot = True):
        image = image.copy()
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        for centroid in centroids:
            cv.circle(image, tuple(centroid.astype('int')), 4, (1.0, 0.0, 0.0), -1);
        if plot:
            plt.imshow(image)
        return image
    
    # the main method for object detection
    def detect(self, inputColor, plot = True):
        croppedRect = self.croppedRect
        maxNumObjects = self.maxNumObjects
        croppedInputColor = self.cropImage(inputColor, croppedRect) 
        centroids,_ = self.centroidDetection(croppedInputColor, maxNumObjects)
        image = self.detectionPlot(croppedInputColor, centroids, plot)
        coordinates = np.concatenate((centroids), axis = 1)
        coordinates = self.reconstructor.reconstruct(croppedRect[0] + coordinates[:,0], croppedRect[1] + coordinates[:,1])
        return coordinates, centroids, croppedInputColor, image
    
