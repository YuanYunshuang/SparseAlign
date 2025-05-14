
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsLineItem
from PyQt5 import QtCore

from cosense3d.utils.box_utils import *
from cosense3d.dataset.toolkit.cosense import csColors
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs

CSCOLORS = (np.array([csColors[k] for k in cs.OBJ_LIST]) / 255.).tolist()

BOX_COLORs = {
    'inactive': CSCOLORS,
    'highlight': (0., 1, 1, 1),
    'active': (0.9, 0, 1, 1),
    'local_gt': (1, 1, 0, 1),
    'global_gt': (0, 1, 0, 1),
    'gt': (0, 1, 0, 1),
    'det': (1, 0, 0, 1),
    'pred': (1, 0, 1, 1),
    'successor': (0, 0.5, 1, 1),
    'successor_gt': (0, 1, 1, 1)
}

pens = {
    'yellow_dashed': pg.mkPen('y', width=1, style=QtCore.Qt.DashLine),
    'yellow_solid': pg.mkPen('y', width=1, style=QtCore.Qt.SolidLine),
    'virtual': pg.mkPen(color=(0, 0, 0, 0), width=1),
}


class MeshBoxItem(gl.GLMeshItem):
    def __init__(self, size=(1, 1, 1), color=(0.0, 1.0, 0.0, 0.25)):
        l, w, h = size
        verts = [
            [0, 0, 0],
            [l, 0, 0],
            [l, 0, h],
            [0, 0, h],
            [0, w, 0],
            [l, w, 0],
            [l, w, h],
            [0, w, h]
        ]
        verts = np.array(verts)

        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [1, 5, 6],
            [1, 6, 2],
            [5, 4, 7],
            [5, 7, 6],
            [4, 0, 3],
            [4, 3, 7],
            [3, 2, 6],
            [3, 6, 7],
            [0, 4, 5],
            [0, 5, 1]
        ]
        faces = np.array(faces)

        normals = np.array([
            [0, -1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, -1],
            [0, 0, 1]
        ])

        colors = [color] * len(faces)

        meshdata = gl.MeshData(vertexes=verts, faces=faces, faceColors=colors)
        super().__init__(meshdata=meshdata, shader='balloon', glOptions='translucent')


class LineBoxItem(gl.GLLinePlotItem):
    ids = set()  # TODO: need to be initialized by labeled data in the current scenario
    id_ptr = 0
    def __init__(self,
                 box,
                 status='inactive',
                 show_direction=False,
                 last_pose=None,
                 line_width=1.):
        """
        :param box: ([id], type_id, x, y, z, l, w, h, roll, pitch, yaw)
        :param color:

                4 -------- 5             ^ z
               /|         /|             |
              7 -------- 6 .             |
              | |        | |             | . x
              . 0 -------- 1             |/
              |/         |/              +-------> y
              3 -------- 2
        """
        id = None
        box_score = None
        if len(box) == 11:
            id = int(box[0])
            type_id = int(box[1])
            box = box[2:]
        elif len(box) == 10:
            type_id = int(box[0])
            box = box[1:]
        elif len(box) == 12:
            id = int(box[0])
            type_id = int(box[1])
            box_score = box[-1]
            box = box[2:-1]
        else:
            raise NotImplementedError
        vertices = np.zeros((12, 3))
        vertices[:8] = boxes_to_corners_3d(np.array([box]))[0]
        if show_direction:
            # -----
            # |   |---- direction on top
            # -----
            top_center = np.mean(vertices[4:], axis=0)
            top_front = np.mean(vertices[[4, 5]], axis=0)
            top_ff = top_front * 2 - top_center
            vertices[8] = top_front
            vertices[9] = top_ff
        if last_pose is not None:
            #                                       -----
            # last pose on bottom of the boxe  o----|   |
            #                                       -----
            assert len(last_pose) == 3
            bottom_center = np.mean(vertices[:4], axis=0)
            last_pose[2] = bottom_center[2]  # set last pose z to ground
            vertices[10] = np.array(last_pose)
            vertices[11] = np.array(bottom_center)

        self.vertices = vertices

        # Define the edges of the box
        edges = [
            [0, 1],  # front-bottom
            [1, 5],  # front-right
            [5, 4],  # front-top
            [4, 0],  # front-left
            [0, 3],  # left-bottom
            [1, 2],  # right-bottom
            [5, 6],  # right-top
            [4, 7],  # left-top
            [3, 2],  # back-bottom
            [2, 6],  # back-right
            [6, 7],  # back-top
            [7, 3],  # back-left
        ]
        if show_direction:
            edges.append([8, 9])
        if last_pose is not None:
            edges.append([10, 11])
        self.edges = np.array(edges)

        vertices_pairs = self.vertices[self.edges.flatten()]

        while id is None:
            if LineBoxItem.id_ptr not in LineBoxItem.ids:
                id = LineBoxItem.id_ptr
            else:
                LineBoxItem.id_ptr += 1
        self.id = id
        self.typeid = type_id
        LineBoxItem.ids.add(id)

        super().__init__(pos=vertices_pairs,
                         color=self.color(status),
                         width=line_width,
                         mode='lines',
                         glOptions='opaque')

    def to_center(self):
        """Convert box to center format"""
        transform = self.transform().matrix()
        corners = (transform[:3, :3] @ self.vertices[:8].T) + transform[:3, 3:]
        box_center = corners_to_boxes_3d(corners.T[None, :])
        return box_center[0]

    def activate(self):
        self.setData(color=BOX_COLORs['active'], width=2.0)

    def deactivate(self):
        self.setData(color=BOX_COLORs['inactive'][self.typeid] + [0.5])

    def highlight(self):
        self.setData(color=BOX_COLORs['highlight'], width=2.0)

    @property
    def isActive(self):
        return self.color == BOX_COLORs['active']

    def color(self, status):
        if status in ['inactive']:
            return BOX_COLORs[status][self.typeid] + [0.5]
        else:
            return BOX_COLORs[status]


class LineItem(QGraphicsLineItem):
    def __init__(self, line, parent=None):
        super().__init__(parent)
        self.inactive_pen = pens['yellow_dashed']
        self.active_pen = pens['yellow_solid']
        self.setLine(*line)
        self.setPen(self.inactive_pen)
        self.setZValue(5)
        self.active = False

    def hoverEvent(self, event):
        if event.isExit():
            self.setPen(self.inactive_pen)
            self.active = False
        else:
            self.setPen(self.active_pen)
            self.active = True


class RectangleItem(QGraphicsRectItem):
    def __init__(self, rect):
        super().__init__(*rect)
        self.setPen(pens['virtual'])
        self.setZValue(0)
        self.active = False

    def hoverEvent(self, event):
        if event.isExit():
            self.setPen(pens['virtual'])
            self.active = False
        else:
            pos = event.pos()
            if abs(pos.x()) < 0.3 and abs(pos.y()) < 0.3:
                self.setPen(pens['yellow_solid'])
                self.active = True



if __name__ == "__main__":
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()

    boxItem = LineBoxItem(
        box=[-5, 8, -1, 4, 3, 2, 0, 0, 0],
        show_direction=True
    )
    w.addItem(boxItem)

    app.exec_()



