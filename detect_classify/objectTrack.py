from object_vision_sensing.detect_classify.objectClassify import objectDetectClassifier
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
import matplotlib.pyplot as plt
    
class nearestTracker(object):
    def __init__(self):
        self.frameNum = 0
        self.storedCoordinates = None
        self.IDs = None
        self.neigh = NearestNeighbors(n_neighbors=1)
    def track(self, coordinates):
        if self.storedCoordinates is None:
            self.storedCoordinates = coordinates
            self.IDs = [ID for ID in range(len(coordinates))]
        else:
            self.neigh.fit(self.storedCoordinates)
            neighbors = self.neigh.kneighbors(coordinates, return_distance=False)
            IDs = [self.IDs[neighbors[i][0]] for i in range(len(coordinates))]
            self.storedCoordinates = coordinates
            self.IDs = IDs
       
        self.frameNum = self.frameNum + 1
        return self.IDs
            
class objectDetectTracker(object):
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker
    
    def setCroppedRect(self, croppedRect):
        self.detector.setCroppedRect(croppedRect)
    
    def setReconstructor(self, cameraIntrinsics = None, rotation = None, transition = None):
        self.detector.setReconstructor(cameraIntrinsics, rotation, transition)
    
    def detectionPlot(self, image, centroids, labels, plot = True):
        image = self.detector.detectionPlot(image, centroids, plot=False)
        font = cv.FONT_HERSHEY_SIMPLEX
        # fontScale 
        fontScale = 0.5
        # Blue color in RGB 
        color = (0.0, 0.0, 1.0) 
        # Line thickness of 2 px 
        thickness = 2
        for centroidId, centroid in enumerate(centroids):
            label = str(labels[centroidId]['id'])
            cv.putText(image, label, (int(centroid[0]-3), int(centroid[1]-10)), font, fontScale, color, thickness, cv.LINE_AA)
        if plot:
            plt.imshow(image)
        return image

    def detect(self, inputColor, inputDepth, plot = True):
        coordinates, centroids, croppedInputColor, _ = self.detector.detect(inputColor, plot = False)
        IDs = self.tracker.track(coordinates)
        labels = [{'id': IDs[coordinateId]} for coordinateId in range(len(coordinates))]
        dic = {tuple(coordinate): labels[coordinateId] for coordinateId, coordinate in enumerate(coordinates)}
        image = self.detectionPlot(croppedInputColor, centroids, labels, plot)
        return dic, coordinates, centroids, croppedInputColor, labels, image
        
