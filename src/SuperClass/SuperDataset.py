# encoding: utf-8


class SuperDatasetManager(object):

    def __init__(self):
        pass

    def readDataset(self, txtPath):
        raise NotImplementedError('Virtual Basic Class')

    def getNextBatch(self, batchSize=0):
        raise NotImplementedError('Virtual Basic Class')

    def getSampleNum(self):
        raise NotImplementedError('Virtual Basic Class')

    # def showImage(self, index=0):
    #     raise NotImplementedError('Virtual Basic Class')

    def showBbox(self, index=0):
        raise NotImplementedError('Virtual Basic Class')


class SuperDataset(object):

    def __init__(self):
        self.testset = SuperDatasetManager()
        self.trainset = SuperDatasetManager()

    def readDataset(self):
        raise NotImplementedError('Virtual Basic Class')


def main():
    superDatasetManager = SuperDatasetManager()
    superDatasetManager.readDataset('a')

if __name__ == '__main__':
    main()
