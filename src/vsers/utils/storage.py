import pickle
import os


class Pickling(object):
    def __init__(self, filename='pickling', foldername='store/'):
        self.filename = filename
        self.foldername = foldername
        self.num = 0
        self.dic = {}

    def store(self, **kwargs):
        self.dic[self.num] = kwargs
        self.num = self.num + 1

    def save(self):
        with open(os.path.join(self.foldername, self.filename), 'wb') as pickling_file:
            pickle.dump(self.dic, pickling_file)
