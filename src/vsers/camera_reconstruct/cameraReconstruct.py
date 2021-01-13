import numpy as np


class cameraReconstructor(object):

    def __init__(self):
        self.cameraIntrinsics = np.identity(3)
        self.rotation = np.identity(3)
        self.transition = np.zeros((3, 1))

    def reset(self, cameraIntrinsics=None, rotation=None, transition=None):
        if cameraIntrinsics is not None:
            self.cameraIntrinsics = cameraIntrinsics
        if rotation is not None:
            self.rotation = rotation
        if transition is not None:
            self.transition = transition

    def reconstruct(self, pixX, pixY):
        cameraIntrinsics = self.cameraIntrinsics
        inputDepth = 1.0
        camX = (pixX -
                cameraIntrinsics[0, 2]) * inputDepth / cameraIntrinsics[0, 0]
        camY = (pixY -
                cameraIntrinsics[1, 2]) * inputDepth / cameraIntrinsics[1, 1]
        camZ = 0.0
        invRotation = np.linalg.inv(self.rotation)
        worldX = invRotation[0, 0] * camX + invRotation[
            0, 1] * camY + invRotation[0, 2] * camZ + self.transition[0, 0]
        worldY = invRotation[1, 0] * camX + invRotation[
            1, 1] * camY + invRotation[1, 2] * camZ + self.transition[1, 0]
        worldZ = invRotation[2, 0] * camX + invRotation[
            2, 1] * camY + invRotation[2, 2] * camZ + self.transition[2, 0]
        coordinate = np.transpose(np.array((worldX, worldY, worldZ)))

        return coordinate
