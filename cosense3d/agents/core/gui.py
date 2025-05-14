
import functools
import os

from PyQt5 import QtCore, QtGui, QtWidgets

from cosense3d.agents.viewer.gl_viewer import GLViewer
from cosense3d.agents.viewer.output_viewer import OutputViewer
from cosense3d.agents.viewer.img_viewer import ImgViewer
from cosense3d.agents.viewer.img_anno3d_viewer import ImgAnno3DViewer


class GUI(QtWidgets.QMainWindow):
    def __init__(self, mode, cfg) -> None:
        super(GUI, self).__init__()
        self.mode = mode
        self.header_height = 30
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.css_dir = os.path.join(path, 'viewer', 'css')
        self.data_keys = [
            'scenario', 'frame',
            'points', 'img', 'bboxes2d', 'lidar2img',
            'global_labels', 'local_labels', 'global_pred_gt',
            'detection', 'detection_local', 'global_pred'
        ]
        self.setupUI(cfg)
        self.setWindowTitle("Cosense3D")

        # Set window size to screen size
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        width, height = screen.width(), screen.height()
        self.setGeometry(0, 0, width, height)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.data = None
        self.colo_mode = 'united'

    def setupUI(self, cfg):
        self.tabs = QtWidgets.QTabWidget()

        self.glViewer0 = GLViewer('MAINVIEW', self)
        self.tabs.addTab(self.glViewer0, 'GLViewer')

        self.img_viewer = ImgViewer(**cfg.get('img_viewer', {}))
        self.tabs.addTab(self.img_viewer, 'ImgViewer')

        self.img_anno3d_viewer = ImgAnno3DViewer(**cfg.get('img_anno3d_viewer', {}))
        self.tabs.addTab(self.img_anno3d_viewer, 'ImgAnno3DViewer')

        self.output_viewer = OutputViewer(**cfg['output_viewer'])
        self.tabs.addTab(self.output_viewer, 'OutputViewer')
        self.data_keys.extend(self.output_viewer.gather_data_keys)

        self.setCentralWidget(self.tabs)
        self.get_toolbar()

    def setRunner(self, runner):
        self.runner = runner

    def initGUI(self):
        # connect all events
        self.connect_events_to_funcs()

    def get_toolbar(self):
        self.toolbar = self.addToolBar("Toolbar")
        self.infos = ['scene', 'frame', 'PCDcolor']
        self.tools = ['start', 'stop', 'step']
        self.visible_objects = ['localDet', 'globalDet', 'localGT', 'globalGT', 'globalPred', 'globalPredGT']

        # add label combo pairs
        for name in self.infos:
            qlabel = QtWidgets.QLabel(f' {name[0].upper()}{name[1:]}:')
            w1 = qlabel.sizeHint().width()
            qlabel.setMinimumWidth(w1 + 25)
            qlabel.setMaximumWidth(w1 + 50)
            qcombo = QtWidgets.QComboBox()
            w2 = qcombo.sizeHint().width()
            qcombo.setMinimumWidth(w2 + 25)
            qcombo.setMaximumWidth(w2 + 50)
            if name=='PCDcolor':
                qcombo.addItem('united')
                qcombo.addItem('height')
                qcombo.addItem('cav')
                qcombo.addItem('time')
            else:
                qcombo.addItem('---------')
            setattr(self, f'label_{name}', qlabel)
            setattr(self, f'combo_{name}', qcombo)
            setattr(self, f'cur_{name}', None)

            self.toolbar.addWidget(getattr(self, f'label_{name}'))
            self.toolbar.addWidget(getattr(self, f'combo_{name}'))

        for name in self.tools:
            bname = f'{name[0].upper()}{name[1:]}'
            qbutton = QtWidgets.QToolButton()
            qbutton.setText(bname)
            # qbutton.setIcon(QtGui.QIcon(f"./interface/ui/icons/{name}.png"))
            w = qbutton.sizeHint().width() + 1
            qbutton.setMaximumWidth(w)
            setattr(self, f'button_{name}', qbutton)
            self.toolbar.addWidget(getattr(self, f'button_{name}'))

        for name in ['glcolor'] + self.visible_objects:
            bname = f'{name[0].upper()}{name[1:]}'
            qbutton = QtWidgets.QPushButton()
            qbutton.setText(bname)
            w = qbutton.sizeHint().width() + 1
            qbutton.setMaximumWidth(w)
            setattr(self, f'button_{name}', qbutton)
            self.toolbar.addWidget(getattr(self, f'button_{name}'))

        for name in self.visible_objects:
            setattr(self, f"{name.lower()}_visible", False)

        self.button_glcolor.setStyleSheet("background-color: black; color: white")

    def change_visible(self, name):
        button = getattr(self, f'button_{name}')
        current_color = button.palette().button().color()

        if current_color != QtGui.QColor('lightblue'):
            button.setStyleSheet("background-color: lightblue")
            setattr(self, f"{name.lower()}_visible", True)
        else:
            button.setStyleSheet("background-color: #efefef")
            setattr(self, f"{name.lower()}_visible", False)
        self.refresh()

    def change_glcolor(self):
        button = self.button_glcolor
        current_color = button.palette().button().color()
        if current_color == QtGui.QColor('black'):
            button.setStyleSheet("background-color: white; color: black")
            self.glViewer0.setBackgroundColor('w')
        else:
            button.setStyleSheet("background-color: black; color: white")
            self.glViewer0.setBackgroundColor('k')
        self.refresh()

    def change_color_mode(self):
        self.colo_mode = self.combo_PCDcolor.currentText()
        self.refresh()

    def connect_events_to_funcs(self):
        self.combo_PCDcolor.currentIndexChanged.connect(self.change_color_mode)
        self.button_step.clicked.connect(self.step)
        self.button_start.clicked.connect(self.start)
        self.button_stop.clicked.connect(self.stop)
        self.tabs.currentChanged.connect(self.refresh)
        self.button_glcolor.clicked.connect(self.change_glcolor)
        for name in self.visible_objects:
            if getattr(self, f"{name.lower()}_visible"):
                self.change_visible(name)
            getattr(self, f'button_{name}').clicked.connect(
                functools.partial(self.change_visible, name=name))

    def step(self):
        self.runner.step()
        self.data = self.runner.vis_data(self.data_keys)
        self.refresh()
        if self.runner.iter == self.runner.total_iter:
            self.timer.stop()

    def refresh(self):
        active_widget = self.tabs.currentWidget()
        if self.data is not None:
            visible_keys = [k for k in self.visible_objects if getattr(self, f"{k.lower()}_visible")]
            active_widget.refresh(self.data, visible_keys=visible_keys, color_mode=self.colo_mode)
            scene = list(self.data['scenario'].values())[0]
            frame = list(self.data['frame'].values())[0]
            # todo adapt scenario and frame selection
            self.combo_frame.clear()
            self.combo_frame.addItem(frame)
            self.combo_scene.clear()
            self.combo_scene.addItem(scene)

    def start(self):
        self.timer.start(300)  # Trigger the animate method every 100ms

    def stop(self):
        self.timer.stop()






