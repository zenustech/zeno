import os
import math
import time
import numpy as np

from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *

from . import zenvis
from .dialog import RecordVideoDialog
from .camera_keyframe import CameraKeyframeWidget

from ..utils import asset_path
from ..editor import locale

from typing import Tuple

from ..keyframe_editor.frame_curve_editor import CurveWindow
from ..keyframe_editor.curve_canvas import ControlPoint

class CameraControl:
    '''
    MMB to orbit, Shift+MMB to pan, wheel to zoom
    '''
    def __init__(self):
        self.mmb_pressed = False
        self.theta = 0.0
        self.phi = 0.0
        self.last_pos = (0, 0)
        self.center = (0.0, 0.0, 0.0)
        self.ortho_mode = False
        self.fov = 45.0
        self.radius = 5.0
        self.res = (1, 1)

        self.update_perspective()

    def reset(self):
        self.center = (0.0, 0.0, 0.0)
        self.radius = 5.0
        zenvis.core.clearCameraControl()
        self.update_perspective()

    def mousePressEvent(self, event):
        if not (event.buttons() & Qt.MiddleButton):
            return

        self.last_pos = event.x(), event.y()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MiddleButton):
            return
        ratio = QApplication.desktop().devicePixelRatio()

        x, y = event.x(), event.y()
        dx, dy = x - self.last_pos[0], y - self.last_pos[1]
        dx *= ratio / self.res[0]
        dy *= ratio / self.res[1]

        shift_pressed = bool(event.modifiers() & Qt.ShiftModifier)

        if shift_pressed:
            cos_t = np.cos(self.theta)
            sin_t = np.sin(self.theta)
            cos_p = np.cos(self.phi)
            sin_p = np.sin(self.phi)
            back = np.array([ cos_t * sin_p, sin_t, -cos_t * cos_p])
            up = np.array([-sin_t * sin_p, cos_t, sin_t * cos_p])
            right = np.cross(up, back)
            up = np.cross(back, right)
            right /= np.linalg.norm(right)
            up /= np.linalg.norm(up)
            delta = right * dx + up * dy
            center = np.array(self.center)
            center = center + delta * self.radius
            self.center = tuple(center)
        else:
            self.theta -= dy * np.pi
            self.phi += dx * np.pi

        self.last_pos = x, y

        self.update_perspective()

    def update_perspective(self):
        cx, cy, cz = self.center
        zenvis.status['perspective'] = (cx, cy, cz,
                self.theta, self.phi, self.radius,
                self.fov, self.ortho_mode)
        zenvis.status['resolution'] = self.res

    def wheelEvent(self, event):
        dy = event.angleDelta().y()
        scale = 0.89 if dy >= 0 else 1 / 0.89

        shift_pressed = bool(event.modifiers() & Qt.ShiftModifier)
        if shift_pressed:
            self.fov /= scale
        self.radius *= scale

        self.update_perspective()

    def set_keyframe(self, keyframe):
        f = keyframe
        self.center = (f[0], f[1], f[2])
        self.theta = f[3]
        self.phi = f[4]
        self.radius = f[5]
        self.fov = f[6]
        self.ortho_mode = f[7]

class ViewportWidget(QGLWidget):
    def __init__(self, parent=None):
        fmt = QGLFormat()
        nsamples = os.environ.get('ZEN_MSAA')
        if not nsamples:
            nsamples = 16
        else:
            nsamples = int(nsamples)
        nsamples = 1
        fmt.setSamples(nsamples)
        fmt.setVersion(3, 0)
        fmt.setProfile(QGLFormat.CoreProfile)
        super().__init__(fmt, parent)

        self.camera = CameraControl()
        zenvis.camera_control = self.camera
        self.record_path = None
        self.record_res = None

        zenvis.core.set_num_samples(nsamples)

    def initializeGL(self):
        zenvis.initializeGL()

    def resizeGL(self, nx, ny):
        # ratio = QApplication.desktop().devicePixelRatio()
        self.camera.res = (nx, ny)
        self.camera.update_perspective()

    def paintGL(self):
        zenvis.paintGL()
        self.check_record()

    def check_record(self):
        f = zenvis.get_curr_frameid()
        if self.record_path and f <= self.frame_end:
            old_res = self.camera.res
            self.camera.res = self.record_res
            self.camera.update_perspective()
            zenvis.recordGL(self.record_path)
            self.camera.res = old_res
            self.camera.update_perspective()
            if f == self.frame_end:
                self.parent_widget.record_video.finish_record()

    def on_update(self):
        self.updateGL()

@eval('lambda x: x()')
def _():
    for name in ['mousePressEvent', 'mouseMoveEvent', 'wheelEvent']:
        def closure(name):
            oldfunc = getattr(ViewportWidget, name)
            def newfunc(self, event):
                getattr(self.camera, name)(event)
                oldfunc(self, event)
            setattr(ViewportWidget, name, newfunc)
        closure(name)


class QDMDisplayMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('Display')

        action = QAction('Show Grid', self)
        action.setCheckable(True)
        action.setChecked(True)
        self.addAction(action)

        action = QAction('Background Color', self)
        self.addAction(action)

        action = QAction('Set Light', self)
        self.addAction(action)

        self.addSeparator()

        action = QAction('Smooth Shading', self)
        action.setCheckable(True)
        action.setChecked(False)
        self.addAction(action)

        action = QAction('Normal Check', self)
        action.setCheckable(True)
        action.setChecked(False)
        self.addAction(action)

        action = QAction('Wireframe', self)
        action.setCheckable(True)
        action.setChecked(False)
        self.addAction(action)

        self.addSeparator()

        action = QAction('Camera Keyframe', self)
        self.addAction(action)

        self.addSeparator()

        action = QAction('Use English', self)
        action.setCheckable(True)
        with open(asset_path('language.txt')) as f:
            lang = f.read()
            action.setChecked(lang == 'en')
        self.addAction(action)

        self.addSeparator()

        action = QAction('Reset Camera', self)
        action.setShortcut(QKeySequence('Shift+C'))
        self.addAction(action)

def get_env_tex_names():
    ns = os.listdir('assets/sky_box')
    return list(ns)

env_texs = get_env_tex_names()

class QDMEnvTexMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('EnvTex')

        for name in env_texs:
            action = QAction(name, self)
            self.addAction(action)

class QDMRecordMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('Record')

        action = QAction('Screenshot', self)
        action.setShortcut(QKeySequence('F12'))
        self.addAction(action)

        action = QAction('Record Video', self)
        action.setShortcut(QKeySequence('Shift+F12'))
        self.addAction(action)

light_channel_names = [
    'dir_x',
    'dir_y',
    'dir_z',
    'height',
    'softness',
    'tint_r',
    'tint_g',
    'tint_b',
]

class LightKeyframe:
    def __init__(self):
        super().__init__()
        self.frame = 0
        self.dir: Tuple[float, float, float] = (1, 1, 0)
        self.height: float = 1000.0
        self.softness: float = 1.0
        self.tint: Tuple[float, float, float] = (0.2, 0.2, 0.2)
        self.color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.intensity: float = 10.0
        self.scale: float = 1.0
        self.enable: float = 1.0

class SetLightDialog(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('LightsWindow')
        self.initUI()

        self.lights = zenvis.status['lights']
        self.new_light_channel()
        self.curve_editor = CurveWindow(self.lights[0])

    def initUI(self):
        self.keyframe_btn = QPushButton('Keyframe')
        self.keyframe_btn.clicked.connect(self.keyframe)

        self.edit_btn = QPushButton('Edit')
        self.edit_btn.clicked.connect(self.edit)

        self.add_light_btn = QPushButton('Add')
        self.add_light_btn.clicked.connect(self.add_light)

        self.remove_light_btn = QPushButton('Remove')
        self.remove_light_btn.clicked.connect(self.remove_light)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.update_item)

        self.enable_checkbox = QCheckBox('Enable')
        self.enable_checkbox.setCheckState(Qt.Checked)
        self.enable_checkbox.stateChanged.connect(self.setLight)

        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setMinimum(0)
        self.phi_slider.setMaximum(100)
        self.phi_slider.valueChanged.connect(self.setLight)

        self.theta_slider = QSlider(Qt.Horizontal)
        self.theta_slider.setMinimum(0)
        self.theta_slider.setMaximum(100)
        self.theta_slider.valueChanged.connect(self.setLight)

        self.height_spinbox = QLineEdit('0')
        self.height_spinbox.textEdited.connect(self.setLight)

        self.softness_spinbox = QLineEdit('1')
        self.softness_spinbox.textEdited.connect(self.setLight)

        self.scale_spinbox = QLineEdit('1')
        self.scale_spinbox.textEdited.connect(self.setLight)

        self.shadow_tint_spinbox_r = QDoubleSpinBox()
        self.shadow_tint_spinbox_r.valueChanged.connect(self.setLight)
        self.shadow_tint_spinbox_r.setSingleStep(0.05)
        self.shadow_tint_spinbox_g = QDoubleSpinBox()
        self.shadow_tint_spinbox_g.valueChanged.connect(self.setLight)
        self.shadow_tint_spinbox_g.setSingleStep(0.05)
        self.shadow_tint_spinbox_b = QDoubleSpinBox()
        self.shadow_tint_spinbox_b.valueChanged.connect(self.setLight)
        self.shadow_tint_spinbox_b.setSingleStep(0.05)

        self.color_spinbox_r = QDoubleSpinBox()
        self.color_spinbox_r.valueChanged.connect(self.setLight)
        self.color_spinbox_r.setSingleStep(0.05)
        self.color_spinbox_g = QDoubleSpinBox()
        self.color_spinbox_g.valueChanged.connect(self.setLight)
        self.color_spinbox_g.setSingleStep(0.05)
        self.color_spinbox_b = QDoubleSpinBox()
        self.color_spinbox_b.valueChanged.connect(self.setLight)
        self.color_spinbox_b.setSingleStep(0.05)

        self.light_intensity = QDoubleSpinBox()
        self.light_intensity.valueChanged.connect(self.setLight)
        self.light_intensity.setSingleStep(0.05)
        
        layout = QVBoxLayout()
        layout.addWidget(self.keyframe_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.add_light_btn)
        layout.addWidget(self.remove_light_btn)
        layout.addWidget(self.list)

        layout.addWidget(self.enable_checkbox)

        layout.addWidget(QLabel('Phi'))
        layout.addWidget(self.phi_slider)

        layout.addWidget(QLabel('Theta'))
        layout.addWidget(self.theta_slider)

        layout.addWidget(QLabel('Height'))
        layout.addWidget(self.height_spinbox)

        layout.addWidget(QLabel('Softness'))
        layout.addWidget(self.softness_spinbox)

        layout.addWidget(QLabel('Scale'))
        layout.addWidget(self.scale_spinbox)

        layout.addWidget(QLabel('ShadowTint'))
        layout.addWidget(self.shadow_tint_spinbox_r)
        layout.addWidget(self.shadow_tint_spinbox_g)
        layout.addWidget(self.shadow_tint_spinbox_b)

        layout.addWidget(QLabel('LightColor'))
        layout.addWidget(self.color_spinbox_r)
        layout.addWidget(self.color_spinbox_g)
        layout.addWidget(self.color_spinbox_b)


        layout.addWidget(QLabel('LightIntensity'))
        layout.addWidget(self.light_intensity)

        self.setLayout(layout)

    def paintEvent(self, e):
        super().paintEvent(e)
        count = zenvis.core.getLightCount()
        if self.list.count() != count:
            self.list.clear()
            self.list.addItems(map(str, range(count)))
    
    def add_light(self):
        zenvis.core.addLight()
        self.new_light_channel()
        self.update()

    def remove_light(self):
        index = self.list.currentRow()
        if index == -1:
            return
        zenvis.core.removeLight(index)
        self.update()
    
    def update_item(self):
        index = self.list.currentRow()
        if index == -1:
            return
        l = zenvis.core.getLight(index)

        x, y, z = l[0]
        phi, theta = self.xyz_sphere(x, y, z)
        self.phi_slider.setValue(round(phi * 100))
        self.theta_slider.setValue(round(theta * 100))

        self.height_spinbox.setText(str(l[1]))
        self.softness_spinbox.setText(str(l[2]))

        sr, sg, sb = l[3]

        self.shadow_tint_spinbox_r.setValue(sr)
        self.shadow_tint_spinbox_g.setValue(sg)
        self.shadow_tint_spinbox_b.setValue(sb)

        cr, cg, cb = l[4]
        self.color_spinbox_r.setValue(cr)
        self.color_spinbox_g.setValue(cg)
        self.color_spinbox_b.setValue(cb)

        intensity = l[5]
        self.light_intensity.setValue(intensity)

        scale = str(l[6])
        self.scale_spinbox.setText(scale)

        enable = l[7]
        self.enable_checkbox.setCheckState(Qt.Checked if enable else Qt.Unchecked)

        self.curve_editor.widget_state.data = self.lights[index]
        self.curve_editor.update()

    def sphere_xyz(self, phi, theta):
        phi = phi * math.pi * 2
        theta = theta * math.pi
        x = math.sin(theta) * math.cos(phi)
        y = -math.cos(theta)
        z = math.sin(theta) * math.sin(phi)
        return x, y, z

    def xyz_sphere(self, x, y, z):
        theta = math.acos(-y) / math.pi
        if theta < 0.01 or theta > 0.99:
            return (0, theta)
        x = x / math.sin(theta)
        z = z / math.sin(theta)
        phi = math.atan2(z, x) / math.pi * 0.5
        if phi < 0:
            phi = phi + 1
        return phi, theta

    def setLight(self):
        index = self.list.currentRow()
        if index == -1:
            return
        lkf = self.get_keyframe()
        zenvis.core.setLightData(
            index,
            lkf.dir,
            lkf.height,
            lkf.softness,
            lkf.tint,
            lkf.color,
            lkf.intensity,
            lkf.scale,
            lkf.enable > 0.5,
        )

    def keyframe(self):
        index = self.list.currentRow()
        if index == -1:
            return
        f = zenvis.get_curr_frameid()
        lkf = self.get_keyframe()
        new_frame = {
            'DirX': ControlPoint(f, lkf.dir[0]),
            'DirY': ControlPoint(f, lkf.dir[1]),
            'DirZ': ControlPoint(f, lkf.dir[2]),
            'Height': ControlPoint(f, lkf.height),
            'Softness': ControlPoint(f, lkf.softness),
            'ShadowR': ControlPoint(f, lkf.tint[0]),
            'ShadowG': ControlPoint(f, lkf.tint[1]),
            'ShadowB': ControlPoint(f, lkf.tint[2]),
            'ColorR': ControlPoint(f, lkf.color[0]),
            'ColorG': ControlPoint(f, lkf.color[1]),
            'ColorB': ControlPoint(f, lkf.color[2]),
            'Intensity': ControlPoint(f, lkf.intensity),
            'Scale': ControlPoint(f, lkf.scale),
            'Enable': ControlPoint(f, lkf.enable, 'constant'),
        }
        for n, l in self.lights[index].items():
            count = len(list(filter(lambda k: k.pos.x <= f, l)))
            if l[count - 1].pos.x == f:
                l[count - 1] = new_frame[n]
            else:
                l.insert(count, new_frame[n])
        self.curve_editor.update()

    def edit(self):
        self.curve_editor.show()

    def get_keyframe(self):
        lkf = LightKeyframe()
        phi = self.phi_slider.value() / 100
        theta = self.theta_slider.value() / 100
        lkf.dir = self.sphere_xyz(phi, theta)
        lkf.height = float(self.height_spinbox.text())
        lkf.softness = float(self.softness_spinbox.text())
        lkf.scale = float(self.scale_spinbox.text())
        lkf.enable = 1.0 if self.enable_checkbox.checkState() == Qt.Checked else 0.0
        lkf.tint = (
            self.shadow_tint_spinbox_r.value(),
            self.shadow_tint_spinbox_g.value(),
            self.shadow_tint_spinbox_b.value(),
        )
        lkf.color = (
            self.color_spinbox_r.value(),
            self.color_spinbox_g.value(),
            self.color_spinbox_b.value(),
        )
        lkf.intensity = self.light_intensity.value()
        return lkf

    def new_light_channel(self):
        new_channel = {
            'DirX': [ControlPoint(0, 1)],
            'DirY': [ControlPoint(0, 1)],
            'DirZ': [ControlPoint(0, 0)],
            'Height': [ControlPoint(0, 1000.0)],
            'Softness': [ControlPoint(0, 1.0)],
            'ShadowR': [ControlPoint(0, 0.2)],
            'ShadowG': [ControlPoint(0, 0.2)],
            'ShadowB': [ControlPoint(0, 0.2)],
            'ColorR': [ControlPoint(0, 1)],
            'ColorG': [ControlPoint(0, 1)],
            'ColorB': [ControlPoint(0, 1)],
            'Intensity': [ControlPoint(0, 10.0)],
            'Scale': [ControlPoint(0, 1.0)],
            'Enable': [ControlPoint(0, 1.0, 'constant')],
        }
        self.lights[len(self.lights)] = new_channel

class DisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.menubar = QMenuBar()
        self.menubar.setMaximumHeight(26)
        self.layout.addWidget(self.menubar)

        self.menuDisplay = QDMDisplayMenu()
        self.menuDisplay.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.menuDisplay)

        self.recordDisplay = QDMRecordMenu()
        self.recordDisplay.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.recordDisplay)

        self.envTexDisplay = QDMEnvTexMenu()
        self.envTexDisplay.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.envTexDisplay)

        self.view = ViewportWidget()
        self.view.parent_widget = self
        self.layout.addWidget(self.view)

        self.record_video = RecordVideoDialog(self)
        self.camera_keyframe_widget = CameraKeyframeWidget(self)
        zenvis.camera_keyframe = self.camera_keyframe_widget

        self.set_light_dialog = SetLightDialog()

    def on_update(self):
        self.view.on_update()

    def menuTriggered(self, act):
        name = act.text()
        if name == 'Show Grid':
            checked = act.isChecked()
            zenvis.status['show_grid'] = checked

        elif name == 'Smooth Shading':
            checked = act.isChecked()
            zenvis.core.set_smooth_shading(checked)

        elif name == 'Normal Check':
            checked = act.isChecked()
            zenvis.core.set_normal_check(checked)

        elif name == 'Wireframe':
            checked = act.isChecked()
            zenvis.core.set_render_wireframe(checked)

        elif name == 'Background Color':
            c = QColor.fromRgbF(*zenvis.core.get_background_color())
            c = QColorDialog.getColor(c)
            if c.isValid():
                zenvis.core.set_background_color(
                    c.redF(),
                    c.greenF(),
                    c.blueF(),
                )

        elif name == 'Set Light':
            self.set_light_dialog.show()

        elif name == 'Record Video':
            checked = act.isChecked()
            self.record_video.do_record_video()

        elif name == 'Screenshot':
            self.do_screenshot()

        elif name == 'Camera Keyframe':
            self.camera_keyframe_widget.show()

        elif name == 'Use English':
            checked = act.isChecked()
            with open(asset_path('language.txt'), 'w') as f:
                f.write('en' if checked else 'zh-cn')
            QMessageBox.information(None, 'Zeno', 'Language switched! Please reboot zeno!')

        elif name == 'Reset Camera':
            self.view.camera.reset()

        elif name in env_texs:
            zenvis.core.setup_env_map(name)

    def get_output_path(self, extname):
        dir_path = 'outputs'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        file_name += extname
        path = os.path.join(dir_path, file_name)
        return path

    def do_screenshot(self):
        path = self.get_output_path('.png')
        zenvis.core.do_screenshot(path)

        msg = 'Saved screenshot to {}!'.format(path)
        QMessageBox.information(self, 'Screenshot', msg)

    def sizeHint(self):
        if os.environ.get('ZEN_NOVIEW'):
            return QSize(1200, 0)
        else:
            return QSize(1200, 400)
