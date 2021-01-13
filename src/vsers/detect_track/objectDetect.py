import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt
from camera_reconstruct.cameraReconstruct import cameraReconstructor


class objectDetector(object):

    def __init__(self, maxNumObjects=1):
        self.maxNumObjects = maxNumObjects
        self.reconstructor = cameraReconstructor()
        self.croppedRect = None

    # set the region of interest for cropping
    def set_cropped_rect(self, croppedRect):
        self.croppedRect = croppedRect

    def set_reconstructor(self,
                          cameraIntrinsics=None,
                          rotation=None,
                          transition=None):
        self.reconstructor.reset(cameraIntrinsics=cameraIntrinsics,
                                 rotation=rotation,
                                 transition=transition)

    def plot_color_image(self, colorimg):
        plt.imshow(colorimg)

    def crop_image(self, inputImg, croppedRect):
        if isinstance(croppedRect, type(None)):
            return inputImg
        assert type(croppedRect) == np.ndarray
        [x, y, dx, dy] = croppedRect.astype('int')
        img = inputImg[y:y + dy, x:x + dx]
        return img

    # detect the centroid of the objects in the image, output are the coordinates of the centroids, and a list of opencv contour objects
    def centroid_detection(self, image, maxNumObjects):
        image = image.copy()
        image = image.astype('uint8')
        blurred = cv.GaussianBlur(image, (5, 5), 0)
        thresh = cv.threshold(blurred, 0, 255,
                              cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [cnt for cnt in cnts if cv.contourArea(cnt) > 5.0]
        cnts = sorted(cnts, key=lambda x: -cv.contourArea(x))
        filteredCount = min(len(cnts), maxNumObjects)
        cnts = cnts[:filteredCount]
        centroids = np.zeros((filteredCount, 2), dtype=int)
        for cntId in range(len(cnts)):
            cnt = cnts[cntId]
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids[cntId, 0] = cx
            centroids[cntId, 1] = cy
        return centroids, cnts

    def detection_plot(self, image, centroids, plot=True):
        image = image.copy()
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        for centroid in centroids:
            cv.circle(image, tuple(centroid.astype('int')), 5, (1.0, 0.0, 0.0),
                      -1)
        if plot:
            plt.imshow(image)
        return image

    # the main method for object detection
    def detect(self, inputColor, plot=True):
        croppedRect = self.croppedRect
        maxNumObjects = self.maxNumObjects
        croppedInputColor = self.crop_image(inputColor, croppedRect)
        centroids, _ = self.centroid_detection(croppedInputColor, maxNumObjects)
        image = self.detection_plot(croppedInputColor, centroids, plot)
        coordinates = centroids
        coordinates = self.reconstructor.reconstruct(
            croppedRect[0] + coordinates[:, 0],
            croppedRect[1] + coordinates[:, 1])
        return coordinates, centroids, croppedInputColor, image
