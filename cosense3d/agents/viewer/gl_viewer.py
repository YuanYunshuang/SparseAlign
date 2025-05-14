

from typing import TYPE_CHECKING, List, Tuple, Union

Color4f = Tuple[float, float, float, float]  # type alias for type hinting

import logging
import queue

import numpy as np
from PyQt5.QtCore import Qt, QEvent, QPointF, QRectF
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from PyQt5.QtGui import QPen, QBrush, QColor
import pyqtgraph.opengl as gl
from matplotlib import colormaps
from OpenGL.GL import *
from OpenGL import GLU
from cosense3d.agents.viewer.utils import depth_min
from cosense3d.agents.viewer.items.graph_items import LineBoxItem

SIZE_OF_FLOAT = ctypes.sizeof(ctypes.c_float)
TRANSLATION_FACTOR = 0.03
jet = colormaps['jet']
cav_colors = np.array([
    [0.745, 0.039, 1.000, 1.000],
    [0.039, 0.937, 1.000, 1.000],
    [0.078, 0.490, 0.961, 1.000],
    [0.039, 1.000, 0.600, 1.000],
    [1.000, 0.529, 0.000, 1.000],
    [0.345, 0.039, 1.000, 1.000],
    [0.631, 1.000, 0.039, 1.000],
    [1.000, 0.827, 0.000, 1.000],
])


# Main widget for presenting the point cloud and bounding boxes
class GLViewer(gl.GLViewWidget):

    def __init__(self, name: str, parent=None) -> None:
        super(GLViewer, self).__init__(parent)
        self.setObjectName(name)
        self.controller = None

        self.setCameraPosition(distance=300, elevation=30, azimuth=-90)
        self.pan(0, 0, 0)
        self.draw_axes()

        self.tasks = queue.Queue()

        # point cloud data
        self.pcd = None
        self.boxes = []
        self.local_boxes = {}
        self.pcd_items = {}
        self.visibility = {}

        # drag window control
        self.dragging = False
        self.start_pos = None
        self.end_pos = None

        # box control
        self.rectangle = None  # (pos1, pos2)
        self.center = None  # evt pose
        self.highlight_mode = False
        self.highlighted_item = None
        self.activate_item = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)  # for visualization of depth
        glDepthFunc(GL_LESS)  # drawn if depth is less than the existing depth
        glEnable(GL_BLEND)  # enable transparency
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        super().initializeGL()

        depth_enabled = glGetBooleanv(GL_DEPTH_TEST)
        print('viwer init:', depth_enabled)

    def paintGL(self, region=None, viewport=None, useItemNames=False):
        super().paintGL(region, viewport, useItemNames)
        # self.draw_depth_buffer()
        self.addBox()
        self.paintRect()
        # depth_enabled = glGetBooleanv(GL_DEPTH_TEST)
        # print("paintGL", depth_enabled)

    def draw_axes(self):
        axis = gl.GLAxisItem(size=QtGui.QVector3D(5, 5, 5))
        self.addItem(axis)

    def updatePCDs(self, pcds, color_mode='united', **kwargs):
        self.pcds = pcds
        if color_mode  == 'height':
            points_all = np.concatenate([pcd for pcd in pcds.values()], axis=0)
            global_min = points_all[:, 2].min()
            global_max = points_all[:, 2].max()
        elif color_mode  == 'time':
            points_all = np.concatenate([pcd for pcd in pcds.values()], axis=0)
            global_min = points_all[:, -1].min()
            global_max = points_all[:, -1].max()
        else:
            global_min = None
            global_max = None

        for i, (lidar_id, pcd)in enumerate(pcds.items()):
            if color_mode == 'united':
                colors = [1.0, 1.0, 1.0, 1.0]
            elif color_mode == 'height':
                height_norm = (pcd[:, 2] - global_min) / (global_max - global_min)
                colors = jet(height_norm)
            elif color_mode == 'cav':
                colors = cav_colors[i]
                colors[-1] = 0.5
                colors = colors.reshape(1, 4).repeat(len(pcd), 0)
            elif color_mode == 'time':
                time_norm = (pcd[:, -1] - global_min) / (global_max - global_min)
                colors = jet(time_norm)
            else:
                raise NotImplementedError
            item = gl.GLScatterPlotItem(
                pos=pcd[:, :3], size=2, glOptions='opaque', color=colors
            )
            if lidar_id in self.visibility:
                item.setVisible(self.visibility[lidar_id])
            else:
                self.visibility[lidar_id] = True
            self.pcd_items[lidar_id] = item
            self.addItem(item)

    def updateLabel(self, local_labels, global_labels, local_det, global_det,
                    successor=None, successor_gt=None, predecessor=None):
        self.boxes = []
        if local_labels is not None:
            for agent_id, labels in local_labels.items():
                self.local_boxes[agent_id] = []
                for id, label in labels.items():
                    item = LineBoxItem(box=[id, ] + label, last_pose=None,
                                       status='local_gt', line_width=2)
                    item.setVisible(self.visibility.get(f'{agent_id}.0', True))
                    self.local_boxes[agent_id].append(item)
                    self.addItem(item)
        if global_labels is not None:
            for id, label in global_labels.items():
                prev_label = None if predecessor is None else predecessor[id]
                item = LineBoxItem(box=[id, ] + label, last_pose=prev_label,
                                   status='global_gt', line_width=2)
                self.boxes.append(item)
                self.addItem(item)
        if local_det is not None:
            for agent_id, labels in local_det.items():
                self.local_boxes[agent_id] = []
                for id, label in labels.items():
                    item = LineBoxItem(box=[id, ] + label, last_pose=None,
                                       status='det', line_width=2)
                    item.setVisible(self.visibility.get(f'{agent_id}.0', True))
                    self.local_boxes[agent_id].append(item)
                    self.addItem(item)
        if global_det is not None:
            for id, label in global_det.items():
                item = LineBoxItem(box=[id, ] + label, last_pose=None,
                                   status='det', line_width=2)
                self.boxes.append(item)
                self.addItem(item)
        if successor is not None:
            for id, label in successor.items():
                item = LineBoxItem(box=[id, ] + label, last_pose=None,
                                   status='successor', line_width=2)
                self.boxes.append(item)
                self.addItem(item)
        if successor_gt is not None:
            for id, label in successor_gt.items():
                item = LineBoxItem(box=[id, ] + label, last_pose=None,
                                   status='successor_gt', line_width=2)
                self.boxes.append(item)
                self.addItem(item)

    def updateFrameData(self, pcds,
                        local_label=None,
                        global_label=None,
                        local_det=None,
                        global_det=None,
                        predecessor=None,
                        successor=None,
                        successor_gt=None,
                        pcd_color='united'):
        self.clear()
        self.draw_axes()
        self.updatePCDs(pcds, color_mode=pcd_color)
        self.updateLabel(local_label,
                         global_label,
                         local_det,
                         global_det,
                         successor,
                         successor_gt,
                         predecessor,)
        self.update()

    def refresh(self, data_dict, visible_keys=['globalGT'], color_mode='united', **kwargs):
        pcds = data_dict.get('points', {})
        ego_id = list(data_dict['scenario'].keys())[0]
        local_labels, global_labels, local_det, global_det = None, None, None, None
        global_pred_gt, global_pred = None, None
        if 'globalGT' in visible_keys:
            global_labels = data_dict.get('global_labels', {})
            global_labels = global_labels[ego_id]
        if 'localGT' in visible_keys:
            local_labels = data_dict.get('local_labels', {})
        if pcds is None and global_labels is {} and local_labels is None:
            return

        if 'localDet' in visible_keys:
            if 'detection_local' in data_dict:
                local_det = {k: v.get('labels', {}) for k, v in data_dict['detection_local'].items()}
        if 'globalDet' in visible_keys:
            if 'detection' in data_dict:
                global_det = data_dict.get('detection', {})
            else:
                global_det = data_dict.get('detection_global', {})
            global_det = global_det.get(ego_id, {'labels': {}})['labels']
        if 'globalPredGT' in visible_keys:
            global_pred_gt = data_dict.get('global_pred_gt', {})
            global_pred_gt = global_pred_gt.get(ego_id, {})
        if 'globalPred' in visible_keys:
            global_pred = data_dict.get('global_pred', {})
            global_pred = global_pred.get(ego_id, {'labels': {}})['labels']

        self.updateFrameData(pcds,
                             local_label=local_labels,
                             global_label=global_labels,
                             local_det=local_det,
                             global_det=global_det,
                             successor=global_pred,
                             successor_gt=global_pred_gt,
                             pcd_color=color_mode)

    def addBox(self):
        if self.rectangle is not None:
            world_pos = self.evt_pos_to_world(*self.rectangle)
            self.rectangle = None
            if world_pos is not None:
                box = LineBoxItem([self.controller.curr_box_type] + [0, 0, 0] + [4, 2, 1.7] + [0, 0, 0])
                azi = self.opts['azimuth']
                box.rotate(azi, 0, 0, 1)
                box.translate(*world_pos, False)
                self.boxes.append(box)
                self.addItem(box)
                self.controller.save_frame_labels(self.boxes)
                logging.info("Add box: ", box.id)
        if self.center is not None:
            world_pos = self.evt_pos_to_world(self.center)
            self.center = None
            if world_pos is not None:
                self.controller.track_singleton(world_pos)

    def highlightBox(self, pos):
        w = 30
        h = 30
        x = pos.x() - w / 2
        y = pos.y() - h / 2
        self.removeHeilight()
        items = self.itemsAt((x, y, w, h))
        for item in items:
            if isinstance(item, LineBoxItem):
                item.highlight()
                self.highlighted_item = item
                self.update()
                return

    def removeHeilight(self):
        if self.highlighted_item is not None:
            self.highlighted_item.deactivate()
            self.highlighted_item = None
            self.update()

    def selectHeilight(self):
        # remove previous activate item if exists
        self.removeActivate()
        self.highlighted_item.activate()
        self.activate_item = self.highlighted_item
        self.highlighted_item = None
        self.controller.show_obj_info(self.activate_item)
        self.update()

    def removeActivate(self):
        if self.activate_item is not None:
            self.activate_item.deactivate()
            self.controller.hide_obj_info()
            self.update()

    def mousePressEvent(self, evt: QtGui.QMouseEvent) -> None:
        depth_enabled = glGetBooleanv(GL_DEPTH_TEST)
        print('mousePressEvent:', depth_enabled)
        self.mousePos = evt.pos()
        if evt.button() == Qt.LeftButton and \
                evt.modifiers() == Qt.ShiftModifier:
            logging.debug("mousePress+Shift: drag box")
            self.start_pos = evt.pos()
            self.end_pos = evt.pos()
            self.dragging = True
        elif evt.button() == Qt.LeftButton and \
                self.highlighted_item is not None:
            logging.debug("Select Highlighted box")
            self.selectHeilight()
        elif evt.button() == Qt.LeftButton and not self.highlight_mode:
            self.removeActivate()
        else:
            super().mousePressEvent(evt)

    def mouseDoubleClickEvent(self, evt: QtGui.QMouseEvent) -> None:
        if evt.button() == Qt.LeftButton:
            self.center = evt.pos()
            logging.debug('Double click left mouse button.')
        self.update()

    def mouseMoveEvent(self, evt: QtGui.QMouseEvent) -> None:
        if evt.buttons() == Qt.LeftButton and \
                evt.modifiers() == Qt.ShiftModifier:
            logging.debug("mousePress+Shift+mouseMove")
            if self.dragging:
                self.end_pos = evt.pos()
                self.update()
        elif self.highlight_mode:
            logging.debug("Highlight box")
            self.highlightBox(evt.pos())
        else:
            super().mouseMoveEvent(evt)
            logging.debug("mouseMove-super")

    def mouseReleaseEvent(self, evt: QtGui.QMouseEvent):
        if evt.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.rectangle = (self.start_pos, self.end_pos)
            self.start_pos = None
            self.end_pos = None
            self.update()
        else:
            super().mouseReleaseEvent(evt)

    def keyPressEvent(self, evt: QEvent) -> None:
        if evt.isAutoRepeat():
            return
        if evt.key() == Qt.Key_Shift:
            logging.debug("keyShiftPressed")
            self.key_shift = True
        elif evt.key() == Qt.Key_C:
            logging.debug("keyCressed: highlight mode")
            self.highlight_mode = True
            self.setMouseTracking(True)
        elif evt.key() == Qt.Key_3:
            evt.accept()
            self.controller.last_frame()
        elif evt.key() == Qt.Key_4:
            evt.accept()
            self.controller.next_frame()
        elif evt.key() == Qt.Key_T:
            evt.accept()
            self.controller.track()
        elif evt.key() == Qt.Key_2:
            evt.accept()
            self.controller.next_frame()
            self.controller.track()
        else:
            super().keyPressEvent(evt)

    def keyReleaseEvent(self, event: QEvent) -> None:
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_C:
            logging.debug("key C released:  deactivate highlighted box")
            self.highlight_mode = False
            self.setMouseTracking(False)
            self.removeHeilight()

    def model_pose_to_world(self, x, y, z):
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = self.getViewport()
        world_pos = GLU.gluUnProject(
            x, y, z, modelview, projection, viewport
        )
        return world_pos

    def evt_pos_to_world(self, pos1, pos2=None):
        """
        Args:
            pos1: center pos if pos2 is None else start post of a region
            pos2: end pos of a region
        """
        if pos2 is None:
            pos1 = QtCore.QPoint(pos1.x() - 20, pos1.y() - 20)
            pos2 = QtCore.QPoint(pos1.x() + 20, pos1.y() + 20)
        depths = self.get_region_depth(pos1, pos2)
        valid = depths < 1
        if valid.sum() == 0:
            logging.info("No point found, skip drawing box")
            return None
        else:
            z = depths[valid].mean()
            y = (pos1.y() + pos2.y()) / 2
            x = (pos1.x() + pos2.x()) / 2
            real_y = self.height() - y
            world_pos = self.model_pose_to_world(x, real_y, z)
        return world_pos

    def get_point_depth(self, x, y):
        buffer_size = 201
        center = buffer_size // 2 + 1
        depths = glReadPixels(
            x - center + 1,
            y - center + 1,
            buffer_size,
            buffer_size,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
        )
        z = depths[center][center]  # Read selected pixel from depth buffer

        if z == 1:
            z = depth_min(depths, center)
        return z

    def get_region_depth(self, p1: QtCore.QPoint, p2: QtCore.QPoint) -> np.ndarray:
        """
        Args:
            p1: start point of region.
            p2: end point of region
        """
        buffer_size_x = abs(p2.x() - p1.x())
        buffer_size_y = abs(p2.y() - p1.y())
        x = min(p1.x(), p2.x())
        y = self.height() - max(p1.y(), p2.y())

        # Create a buffer to hold the depth values
        depth_buffer = np.zeros((buffer_size_y, buffer_size_x), dtype=np.float32)

        glReadPixels(
            x, y,
            buffer_size_x,
            buffer_size_y,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            depth_buffer
        )
        depth_buffer = depth_buffer[::-1, :]

        return depth_buffer

    def draw_depth_buffer(self):
        """!!!!
        Remember the depth buffer is only available under paintGL loop.
        Only in this loop the gl context is active.
        """
        # Get the OpenGL extensions string
        depth_enabled = glGetBooleanv(GL_DEPTH_TEST)
        print(depth_enabled)
        # Retrieve the dimensions of the framebuffer
        viewport = glGetIntegerv(GL_VIEWPORT)
        width, height = viewport[2], viewport[3]

        # Create a buffer to hold the depth values
        depth_buffer = np.zeros((height, width), dtype=np.float32)

        # Read the depth buffer into the buffer
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buffer)
        depth_buffer = depth_buffer[::-1, :]

        # Convert the depth buffer to an image
        print("min depth value:", depth_buffer.min())
        depth_image = ((1 - depth_buffer) * 500) * 255
        depth_image = np.repeat(depth_image[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        # Save the image to a file
        import imageio
        imageio.imwrite('/media/hdd/tmp/depth_image.png', depth_image)

    def box(self):
        p1 = self.box_start_pos
        p2 = self.box_end_pos
        new_lines = np.array([
            [p1.x(), p1.y(), p1.z()],
            [p2.x(), p2.y(), p2.z()],
        ])

        # create a GLLinePlotItem for the axes
        line_item = gl.GLLinePlotItem(pos=new_lines, color=QtGui.QColor(255, 0, 0, 255),
                                      width=3)

        # add the axes to the view
        self.addItem(line_item)

    def drawRectangle(self):
        if self.rectItem is None:
            self.rectItem = pg.QtGui.QGraphicsRectItem()
            self.scene.addItem(self.rectItem)
        x1, y1 = self.startPoint.x(), self.startPoint.y()
        x2, y2 = self.endPoint.x(), self.endPoint.y()
        rect = QRectF(QPointF(x1, y1), QPointF(x2, y2))
        pen = QPen(QColor(255, 0, 0))
        brush = QBrush(QColor(0, 0, 0, 0))
        self.rectItem.setPen(pen)
        self.rectItem.setBrush(brush)
        self.rectItem.setRect(rect)

    def removeRectangle(self):
        if self.rectItem is not None:
            self.scene.removeItem(self.rectItem)
            self.rectItem = None
            self.update()

    def paintRect(self):
        if self.dragging:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            # draw the rectangle
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0)))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0, 80)))
            painter.drawRect(self.start_pos.x(),
                             self.start_pos.y(),
                             self.end_pos.x() - self.start_pos.x(),
                             self.end_pos.y() - self.start_pos.y())

            glEnable(GL_DEPTH_TEST)

    def change_visibility(self, key, visible):
        ai, li = key.split('.')
        self.visibility[key] = visible
        self.pcd_items[key].setVisible(visible)
        for item in self.local_boxes[ai]:
            item.setVisible(visible)




