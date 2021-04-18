import cv2 as cv
import skimage.filters
import numpy as np
import matplotlib.pyplot as plt
from vsers.camera_reconstruct.cameraReconstruct import CameraReconstructor
from vsers.detect_track.objectDetect import ObjectDetector
from vsers.edge_detection.filters import RangeFilter, DownSamplingFilter, ContinuousFilter
from vsers.edge_detection.fitting import ExtrapolateFitting


class EdgeDetector(object):

    def __init__(self, sigma=0.6, minimum=None, maximum=None, fit_method='UnivariateSpline'):
        self.reconstructor = CameraReconstructor()
        self.down_sampling_filter = None
        self.range_filter = RangeFilter(minimum=minimum, maximum=maximum)
        self.down_sampling_filter = DownSamplingFilter()
        self.continuous_filter = None
        self.fit = None
        self.fit_method = fit_method
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

    def edge_detection(self, image, method="canny"):
        image = image.copy()
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = image.astype('uint8')
        if method == "canny":
            edgeImage = self.auto_canny(image, sigma=self.sigma)
        elif method == "sobel":
            image = cv.GaussianBlur(image, (15, 15), 0)
            edgeImage = skimage.filters.sobel(image)
        elif method == "roberts":
            image = cv.GaussianBlur(image, (15, 15), 0)
            edgeImage = skimage.filters.roberts(image)
        else:
            raise NotImplementedError("Edging method not implemented")
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

    def detect(self, inputColor, plot=True, filtering=False, continuous_filtering=False, fitting=False, method="canny"):
        croppedRect = self.croppedRect
        croppedInputColor = self.crop_image(inputColor, croppedRect)
        edgePoints, edgeImage = self.edge_detection(croppedInputColor, method=method)
        image = self.detection_plot(croppedInputColor, edgePoints, plot)
        coordinates = edgePoints
        coordinates = self.reconstructor.reconstruct(
            croppedRect[0] + coordinates[:, 0],
            croppedRect[1] + coordinates[:, 1])
        coordinates = self.range_filter.filter(coordinates)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        if filtering:
            x, y, z = self.down_sampling_filter.filter(coordinates)
            coordinates = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1)
        if continuous_filtering:
            self.continuous_filter = ContinuousFilter()
            coordinates = self.continuous_filter.filter(coordinates)
            x = coordinates[:, 0]
            y = coordinates[:, 1]
        if fitting:
            self.fit = ExtrapolateFitting(fit_method=self.fit_method)
            self.fit.fit(x, y)
            self.fit.get_derivatives()

        return coordinates, edgePoints, croppedInputColor, image, edgeImage, self.fit

    @staticmethod
    def gray_to_black_white(img):
        block_size = 105
        img = cv.GaussianBlur(img, (15, 15), 0)
        local_thresh = skimage.filters.threshold_local(img, block_size, offset=10)
        binary_local = img > local_thresh
        return binary_local
