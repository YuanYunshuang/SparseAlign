

import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cosense3d.utils import vislib

matplotlib.use('Qt5Agg')

class ImgAnno3DViewer(FigureCanvasQTAgg):

    def __init__(self, dpi=50):
        self.fig = Figure(dpi=dpi)
        super(ImgAnno3DViewer, self).__init__(self.fig)

    def refresh(self, data, **kwargs):
        if len(data['img']) == 0:
            return
        self.fig.clear()
        n_cavs = len(data['img'])
        n_imgs = len(list(data['img'].values())[0])
        cav_ids = sorted(list(data['img'].keys()))
        for i, cav_id in enumerate(cav_ids):
            if cav_id in data['local_labels']:
                bboxes3d = np.array(list(data['local_labels'][cav_id].values())
                                    )[:, [1, 2, 3, 4, 5, 6, 9]]
            elif cav_id in data['global_labels']:
                bboxes3d = np.array(list(data['global_labels'][cav_id].values())
                                    )[:, [1, 2, 3, 4, 5, 6, 9]]
            else:
                return
            for j in range(n_imgs):
                ax = self.fig.add_subplot(n_cavs, n_imgs, i * n_imgs + j + 1)
                img = data['img'][cav_id][j].astype(np.uint8)
                lidar2img = data['lidar2img'][cav_id][j]
                vislib.draw_3d_points_boxes_on_img(ax, img, lidar2img, boxes=bboxes3d)
        self.draw()
