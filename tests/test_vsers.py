from vsers.detect_track.objectDetect import objectDetector
from vsers.detect_track.objectTrack import nearestTracker, objectDetectTracker
from vsers.edge_detection.edgeDetect import edgeDetector
import numpy as np

def test_object_detect_setup():
    detector = objectDetector()
    tracker = nearestTracker()
    detectTracker = objectDetectTracker(detector, tracker)
    detectTracker.set_cropped_rect(np.array([400,100,400,200]))


def test_edge_detect_setup():
    detector = edgeDetector()
    detector.set_cropped_rect(np.array([0, 502, 1920, 182]))
