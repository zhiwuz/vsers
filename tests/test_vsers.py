from vsers.detect_track.objectDetect import ObjectDetector
from vsers.detect_track.objectTrack import NearestTracker, ObjectDetectTracker
from vsers.edge_detection.edgeDetect import EdgeDetector
import numpy as np

def test_object_detect_setup():
    detector = ObjectDetector()
    tracker = NearestTracker()
    detectTracker = ObjectDetectTracker(detector, tracker)
    detectTracker.set_cropped_rect(np.array([400,100,400,200]))


def test_edge_detect_setup():
    detector = EdgeDetector()
    detector.set_cropped_rect(np.array([0, 502, 1920, 182]))
