# Code source: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
import numpy as np

class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            assert len(data.shape) >= 2
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            assert len(data.shape) >= 2
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n


if __name__=="__main__":
    from tqdm import tqdm
    data = np.random.random((100, 10, 10, 3))
    stats = StatsRecorder()
    for i, d in tqdm(enumerate(data)):
        d = d.reshape(-1, 3)
        stats.update(d)
        if i > 200:
            break
    print(f"Input Means: [{', '.join(['{:.5f}'] * len(stats.mean))}]".format(*stats.mean))
    print(f"Input  Stds: [{', '.join(['{:.5f}'] * len(stats.std))}]".format(*stats.std))