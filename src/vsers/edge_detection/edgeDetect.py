import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from vsers.camera_reconstruct.cameraReconstruct import CameraReconstructor
from vsers.detect_track.objectDetect import ObjectDetector


class EdgeDetector(object):

    def __init__(self, sigma = 0.6):
        self.reconstructor = CameraReconstructor()
        self.croppedRect = None
        self.sigma = sigma

    def set_cropped_rect(self, croppedRect):
        self.croppedRect = croppedRect

    def set_reconstructor(self,
                          cameraIntrinsics=None,
                          rotation=None,
                          transition=None):
        self.reconstructor.reset(cameraIntrinsics=cameraIntrinsics,
                                 rotation=rotation,
                                 transition=transition)
    @staticmethod
    def crop_image(inputImg, croppedRect):
        return ObjectDetector.crop_image(inputImg, croppedRect)

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv.Canny(image, lower, upper)
        # return the edged image
        return edged

    def edge_detection(self, image):
        image = image.copy()
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = image.astype('uint8')
        edgeImage = self.auto_canny(image, sigma=self.sigma)
        index = np.argmax(edgeImage[::-1, :], axis=0)
        edgePoints = np.array(
            [[x, image.shape[0] - 1 - y] for x, y in enumerate(index) if edgeImage[image.shape[0] - 1 - y, x] > 0])

        return edgePoints, edgeImage

    def detection_plot(self, image, edgePoints, plot=True):
        image = image.copy()
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        for edgePoint in edgePoints:
            cv.circle(image, tuple(edgePoint.astype('int')), 2, (255, 0, 0), -1)
        if plot:
            plt.imshow(image)
        return image

        # the main method for edge detection
    def detect(self, inputColor, plot=True):
        croppedRect = self.croppedRect
        croppedInputColor = self.crop_image(inputColor, croppedRect)
        edgePoints, edgeImage = self.edge_detection(croppedInputColor)
        image = self.detection_plot(croppedInputColor, edgePoints, plot)
        coordinates = edgePoints
        coordinates = self.reconstructor.reconstruct(
            croppedRect[0] + coordinates[:, 0],
            croppedRect[1] + coordinates[:, 1])
        return coordinates, edgePoints, croppedInputColor, image, edgeImage
