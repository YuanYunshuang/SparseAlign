

import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cosense3d.utils import vislib

matplotlib.use('Qt5Agg')


class ImgViewer(FigureCanvasQTAgg):

    def __init__(self, dpi=100, mean=None, std=None):
        self.fig = Figure(dpi=dpi)
        super(ImgViewer, self).__init__(self.fig)
        self.mean = np.array(mean) if mean is not None else None
        self.std = np.array(std) if std is not None else None

    def refresh(self, data, **kwargs):
        if len(data['img']) == 0:
            return
        self.fig.clear()
        n_cavs = len(data['img'])
        n_imgs = len(list(data['img'].values())[0])
        cav_ids = sorted(list(data['img'].keys()))
        for i, cav_id in enumerate(cav_ids):
            for j in range(n_imgs):
                ax = self.fig.add_subplot(n_cavs, n_imgs, i * n_imgs + j + 1)
                img = data['img'][cav_id][j]
                if self.std is not None and self.mean is not None:
                    img = img * self.std + self.mean
                img = img.astype(np.uint8)
                if len(data['bboxes2d']) == 0:
                    bboxes2d = None
                else:
                    bboxes2d = data['bboxes2d'][cav_id][j].reshape(-1, 2, 2)
                vislib.draw_2d_bboxes_on_img(img, bboxes2d, ax)
        self.draw()
