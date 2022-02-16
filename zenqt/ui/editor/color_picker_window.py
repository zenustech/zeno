import math, colorsys, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from copy import deepcopy

def color_picker_clamp(x, lower, upper):
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x

preset_colors = [
    (224, 32, 32),
    (250, 100, 0),
    (247, 181, 0),
    (109, 212, 0),
    (68, 215, 182),
    (50, 197, 255),
    (0, 145, 255),
    (98, 54, 255),
    (182, 32, 224),
    (109, 114, 120),
]


class ColorWall(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.p = parent
        self.layout = QGridLayout()
        self.array = []
        for i in range(len(preset_colors)):
            c = preset_colors[i]
            b = QPushButton()
            b.index = i
            b.clicked.connect(self.button_click)
            b.setStyleSheet("background-color:rgb(" + "{},{},{}".format(*c) + ");")
            self.array.append(b)
        for i in range(len(preset_colors)):
            self.layout.addWidget(self.array[i], i // 5, i % 5)
        self.setLayout(self.layout)
    
    def button_click(self):
        i = self.sender().index
        c = preset_colors[i]
        self.p.updatePanel[QColor].emit(QColor(*c))


class ColorEditorWidget(QWidget):

    def __init__(self, color: QColor = QColor(255, 255, 255)):
        super().__init__()
        self.__current_color = color
        self.__initUi(color)

    def __initUi(self, color: QColor):
        self.__colorPreviewWithGraphics = QWidget()
        self.__colorPreviewWithGraphics.setMinimumSize(100, 75)
        self.setColorPreviewWithGraphics()

        self.__rSpinBox = QSpinBox()
        self.__gSpinBox = QSpinBox()
        self.__bSpinBox = QSpinBox()

        self.__rSpinBox.valueChanged.connect(self.__rColorChanged)
        self.__gSpinBox.valueChanged.connect(self.__gColorChanged)
        self.__bSpinBox.valueChanged.connect(self.__bColorChanged)


        spinBoxs = [self.__rSpinBox, self.__gSpinBox, self.__bSpinBox]
        for spinBox in spinBoxs:
            spinBox.setRange(0, 255)
            spinBox.setAlignment(Qt.AlignCenter)
            spinBox.setFont(QFont('Arial', 12))

        lay = QFormLayout()
        lay.addRow('R', self.__rSpinBox)
        lay.addRow('G', self.__gSpinBox)
        lay.addRow('B', self.__bSpinBox)
        lay.setContentsMargins(0, 0, 0, 0)

        colorEditor = QWidget()
        colorEditor.setLayout(lay)

        lay = QVBoxLayout()
        lay.addWidget(self.__colorPreviewWithGraphics)
        lay.addWidget(colorEditor)

        lay.setContentsMargins(0, 0, 0, 0)

        self.setLayout(lay)

        self.setColor(color)

    def setColorPreviewWithGraphics(self):
        self.__colorPreviewWithGraphics.setStyleSheet(f'background-color: {self.__current_color.name()}; ')

    def setColor(self, color: QColor = QColor(255, 255, 255)):
        self.__current_color = color
        self.setColorPreviewWithGraphics()

        # Prevent infinite valueChanged event loop
        self.__rSpinBox.valueChanged.disconnect(self.__rColorChanged)
        self.__gSpinBox.valueChanged.disconnect(self.__gColorChanged)
        self.__bSpinBox.valueChanged.disconnect(self.__bColorChanged)

        r, g, b = self.__current_color.red(), self.__current_color.green(), self.__current_color.blue()

        self.__rSpinBox.setValue(r)
        self.__gSpinBox.setValue(g)
        self.__bSpinBox.setValue(b)

        self.__rSpinBox.valueChanged.connect(self.__rColorChanged)
        self.__gSpinBox.valueChanged.connect(self.__gColorChanged)
        self.__bSpinBox.valueChanged.connect(self.__bColorChanged)

    def __rColorChanged(self, r):
        self.__current_color.setRed(r)
        self.__procColorChanged()

    def __gColorChanged(self, g):
        self.__current_color.setGreen(g)
        self.__procColorChanged()

    def __bColorChanged(self, b):
        self.__current_color.setBlue(b)
        self.__procColorChanged()

    def __procColorChanged(self):
        self.setColorPreviewWithGraphics()
        self.colorChanged(self.__current_color)

    def getCurrentColor(self):
        return self.__current_color


class ColorHueBarWidget(QWidget):

    def __init__(self, color: QColor = QColor(255, 255, 255)):
        super().__init__()
        self.__initUi(color)

    def __initUi(self, color: QColor):
        self.__hue_bar_height = 300
        self.__hue_bar_width = 20
        self.setMinimumSize(self.__hue_bar_width, self.__hue_bar_height)

        self.__selector_width = 15
        self.__selector_moving_range = self.__hue_bar_height-self.__selector_width

        hueFrame = QWidget(self)

        hueBg = QWidget(hueFrame)
        hueBg.setFixedWidth(self.__hue_bar_width)
        hueBg.setMinimumHeight(self.__hue_bar_height)
        hueBg.setStyleSheet(
            "background-color: qlineargradient(spread:pad, "
            "x1:0, y1:1, x2:0, y2:0, "
            "stop:0 rgba(255, 0, 0, 255), stop:0.166 "
            "rgba(255, 0, 255, 255), stop:0.333 "
            "rgba(0, 0, 255, 255), stop:0.5 "
            "rgba(0, 255, 255, 255), stop:0.666 "
            "rgba(0, 255, 0, 255), stop:0.833 "
            "rgba(255, 255, 0, 255), stop:1 "
            "rgba(255, 0, 0, 255));\n")

        self.__selector = QLabel(hueFrame)
        self.__selector.setGeometry(0, 0, self.__hue_bar_width, self.__selector_width)
        self.__selector.setMinimumSize(self.__hue_bar_width, 0)
        self.__selector.setStyleSheet('background-color: white; border: 2px solid #222; ')
        self.__selector.setText("")

        hueFrame.mouseMoveEvent = self.__moveSelectorByCursor
        hueFrame.mousePressEvent = self.__moveSelectorByCursor

        h, s, v = colorsys.rgb_to_hsv(color.redF(), color.greenF(), color.blueF())
        self.__initHueSelector(h)

    def __moveSelectorByCursor(self, e):
        if e.buttons() == Qt.LeftButton:
            pos = e.pos().y() - math.floor(self.__selector_width/2)
            if pos < 0:
                pos = 0
            if pos > self.__selector_moving_range:
                pos = self.__selector_moving_range
            self.__selector.move(QPoint(0, pos))

            h = self.__selector.y() / self.__selector_moving_range * 100
            self.hueChanged(h)

    def __moveSelectorNotByCursor(self, h):
        geo = self.__selector.geometry()

        # Prevent y from becoming larger than minimumHeight
        # if y becomes larger than minimumHeight, selector will be placed out of the bottom boundary.
        y = int(min(self.__selector_moving_range, h * self.minimumHeight()))
        geo.moveTo(0, y)
        self.__selector.setGeometry(geo)

        h = self.__selector.y() / self.__selector_moving_range * 100
        self.hueChangedByEditor(h)

    def __initHueSelector(self, h):
        self.__moveSelectorNotByCursor(h)

    def moveSelectorByEditor(self, h):
        self.__moveSelectorNotByCursor(h)
    
    def hueChanged(self, h):
        pass

    def hueChangedByEditor(self, h):
        pass

class ColorSquareWidget(QWidget):

    def __init__(self, color: QColor = QColor(255, 255, 255)):
        super().__init__()
        self.__initUi(color)

    def __initUi(self, color: QColor):
        self.setMinimumSize(300, 300)

        self.__h, \
        self.__s, \
        self.__l = colorsys.rgb_to_hsv(color.redF(), color.greenF(), color.blueF())

        # Multiply 100 for insert into stylesheet code
        self.__h *= 100

        self.__colorView = QWidget()
        self.__colorView.setStyleSheet(f'''
            background-color: qlineargradient(x1:1, x2:0, 
            stop:0 hsl({self.__h}%,100%,50%), 
            stop:1 #fff);
        ''')

        self.__blackOverlay = QWidget()
        self.__blackOverlay.setStyleSheet('''
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
            stop:0 rgba(0, 0, 0, 0), 
            stop:1 rgba(0, 0, 0, 255));
            width:100%;
        ''')

        self.__blackOverlay.mouseMoveEvent = self.__moveSelectorByCursor
        self.__blackOverlay.mousePressEvent = self.__moveSelectorByCursor

        self.__selector_diameter = 12

        self.__selector = QWidget(self.__blackOverlay)
        self.__selector.setGeometry(math.floor(self.__selector_diameter/2) * -1,
                                    math.floor(self.__selector_diameter/2) * -1,
                                    self.__selector_diameter,
                                    self.__selector_diameter)
        self.__selector.setStyleSheet('''
            background-color: none;
            border: 1px solid white;
            border-radius: 5px;
        ''')

        self.__blackRingInsideSelector = QLabel(self.__selector)
        self.__blackRingInsideSelector_diameter = self.__selector_diameter-2
        self.__blackRingInsideSelector.setGeometry(QRect(1, 1, self.__blackRingInsideSelector_diameter,
                                                               self.__blackRingInsideSelector_diameter))
        self.__blackRingInsideSelector.setStyleSheet('''
            background-color: none;
            border: 1px solid black;
            border-radius: 5px;
        ''')

        self.__blackRingInsideSelector.setText("")
        
        lay = QGridLayout()
        lay.addWidget(self.__colorView, 0, 0, 1, 1)
        lay.addWidget(self.__blackOverlay, 0, 0, 1, 1)
        lay.setContentsMargins(0, 0, 0, 0)

        self.setLayout(lay)

        self.__initSelector()

    def __moveSelectorNotByCursor(self, s, l):
        geo = self.__selector.geometry()
        x = int(self.minimumWidth() * s)
        y = int(self.minimumHeight() - self.minimumHeight() * l)
        geo.moveCenter(QPoint(x, y))
        self.__selector.setGeometry(geo)

    def __initSelector(self):
        self.__moveSelectorNotByCursor(self.__s, self.__l)

    def __moveSelectorByCursor(self, e):
        if e.buttons() == Qt.LeftButton:
            pos = e.pos()
            if pos.x() < 0:
                pos.setX(0)
            if pos.y() < 0:
                pos.setY(0)
            if pos.x() > 300:
                pos.setX(300)
            if pos.y() > 300:
                pos.setY(300)

            self.__selector.move(pos - QPoint(math.floor(self.__selector_diameter/2),
                                              math.floor(self.__selector_diameter/2)))
            
            self.__setSaturation()
            self.__setLightness()

            self.colorChangedHsl(self.__h, self.__s, self.__l)
    
    def changeHue(self, h):
        self.__h = h
        self.__colorView.setStyleSheet(f'''
            border-radius: 5px;
            background-color: qlineargradient(x1:1, x2:0,
            stop:0 hsl({self.__h}%,100%,50%),
            stop:1 #fff);
        ''')
        self.colorChangedHsl(self.__h, self.__s, self.__l)


    def changeHueByEditor(self, h):
        # Prevent hue from becoming larger than 100
        # if hue becomes larger than 100, hue of square will turn into dark.
        self.__h = min(100, h)
        self.__colorView.setStyleSheet(f'''
            border-radius: 5px;
            background-color: qlineargradient(x1:1, x2:0,
            stop:0 hsl({self.__h}%,100%,50%),
            stop:1 #fff);
        ''')

    def __setSaturation(self):
        self.__s = (self.__selector.pos().x()+math.floor(self.__selector_diameter/2)) / self.minimumWidth()

    def getSaturatation(self):
        return self.__s

    def __setLightness(self):
        self.__l = abs(((self.__selector.pos().y()+math.floor(self.__selector_diameter/2)) / self.minimumHeight()) - 1)

    def getLightness(self):
        return self.__l

    def moveSelectorByEditor(self, s, l):
        self.__moveSelectorNotByCursor(s, l)

    def colorChangedHsl(self, h, s, l):
        pass

class ColorPickerWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.p = parent
        self.__initUi(color=QColor(255, 255, 255))

        self.p.updatePanel[QColor].connect(self.setColor)

    def __initUi(self, color: QColor):
        self.__colorSquareWidget = ColorSquareWidget(color)
        self.__colorSquareWidget.colorChangedHsl = self.__colorChanged

        self.__colorHueBarWidget = ColorHueBarWidget(color)
        self.__colorHueBarWidget.hueChanged = self.__hueChanged
        self.__colorHueBarWidget.hueChangedByEditor = self.__hueChangedByEditor

        self.__colorEditorWidget = ColorEditorWidget(color)
        self.__colorEditorWidget.colorChanged = self.__colorChangedByEditor

        lay = QHBoxLayout()
        lay.addWidget(self.__colorSquareWidget)
        lay.addWidget(self.__colorHueBarWidget)
        lay.addWidget(self.__colorEditorWidget)

        mainWidget = QWidget()
        mainWidget.setLayout(lay)
        lay.setContentsMargins(0, 0, 0, 0)

        self.setLayout(lay)

    def __hueChanged(self, h):
        self.__colorSquareWidget.changeHue(h)
        
    def __hueChangedByEditor(self, h):
        self.__colorSquareWidget.changeHueByEditor(h)
        
    def hsv2rgb(self, h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
        
    def __colorChanged(self, h, s, l):
        r, g, b = self.hsv2rgb(h / 100, s, l)
        color = QColor(r, g, b)
        self.p.updateBar[QColor].emit(color)
        self.__colorEditorWidget.setColor(color)

    def __colorChangedByEditor(self, color: QColor):
        self.p.updateBar[QColor].emit(color)
        h, s, v = colorsys.rgb_to_hsv(color.redF(), color.greenF(), color.blueF())
        self.__colorHueBarWidget.moveSelectorByEditor(h)
        self.__colorSquareWidget.moveSelectorByEditor(s, v)

    def getCurrentColor(self):
        return self.__colorEditorWidget.getCurrentColor()
    
    def setColor(self, c):
        self.__colorChangedByEditor(c)
        self.__colorEditorWidget.setColor(c)

 
black_body_color_ramps = [
    [0,     [0, 0, 0]],
    [0.33,  [1, 0, 0]],
    [0.66,  [1, 1, 0]],
    [1,     [1, 1, 1]],
]
grayscale_color_ramps = [
    [0,     [0, 0, 0]],
    [1,     [1, 1, 1]],
]
infra_red_color_ramps = [
    [0,     [0.2, 0, 1]],
    [0.25,  [0, 0.85, 1]],
    [0.5,   [0, 1, 0.1]],
    [0.75,  [0.95, 1, 0]],
    [1,     [1, 0, 0]],
]
two_tone_color_ramps = [
    [0,     [0, 1, 1]],
    [0.49,  [0, 0, 1]],
    [0.5,   [1, 1, 1]],
    [0.51,  [1, 0, 0]],
    [1,     [1, 1, 0]],
]
white_to_red_color_ramps = [
    [0,     [1, 1, 1]],
    [1,     [1, 0, 0]],
]

class BurningWidget(QWidget):
  
    def __init__(self, updateBar, updatePanel, color_ramps):      
        super().__init__()
        self.select_index = 0
        self.pressed = False
        self.handler_w = 10
        self.height_offset = 3
        self.color_ramps = color_ramps
        
        self.initUI()
        updateBar[QColor].connect(self.change_current_bar_color)
        self.updatePanel = updatePanel
    
    def remove(self):
        if len(self.color_ramps) == 2:
            return
        self.color_ramps.pop(self.select_index)
        if self.select_index == len(self.color_ramps):
            self.select_index -= 1
        self.update()

    def add(self):
        if self.select_index == len(self.color_ramps) - 1:
            return
        p = self.color_ramps[self.select_index]
        n = self.color_ramps[self.select_index + 1]
        self.color_ramps.insert(self.select_index + 1, [
            (p[0] + n[0]) / 2,
            (
                (p[1][0] + n[1][0]) / 2,
                (p[1][1] + n[1][1]) / 2,
                (p[1][2] + n[1][2]) / 2,
            ),
        ])
        self.update()

    def set_preset(self, name):
        name_map = {
            'BlackBody': black_body_color_ramps,
            'Grayscale': grayscale_color_ramps,
            'InfraRed': infra_red_color_ramps,
            'TwoTone': two_tone_color_ramps,
            'WhiteToRed': white_to_red_color_ramps,
        }
        self.color_ramps = deepcopy(name_map[name])
        self.select_index = 0
        self.update()
        
    def initUI(self):
        self.setMinimumSize(1, 30)
        self.value = 75
        self.num = [75, 150, 225, 300, 375, 450, 525, 600, 675]
 
 
    def setValue(self, value):
        self.value = value

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def drawWidget(self, qp):
        size = self.size()
        w = size.width()
        h = size.height()
        
        qp.setPen(Qt.NoPen)
        grad = QLinearGradient(0, 0, w, 0)
        for f, (r, g, b) in self.color_ramps:
            grad.setColorAt(f, QColor(int(r * 255), int(g * 255), int(b * 255)))
        brush = QBrush(grad)
        qp.setBrush(brush)
        qp.drawRect(0, self.height_offset, w, h - 2 * self.height_offset)

        # frame 
        pen = QPen(QColor(20, 20, 20), 1, 
            Qt.SolidLine)
            
        qp.setPen(pen)
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(0, self.height_offset, w-1, h - 1 - 2 * self.height_offset)

        for i in range(len(self.color_ramps)):
            f, (r, g, b) = self.color_ramps[i]
            self.drawHandler(qp, f, QColor(int(r * 255), int(g * 255), int(b * 255)), self.select_index == i)
    
    def drawHandler(self, qp, f, c, selected):
        size = self.size()
        w = size.width() - self.handler_w
        h = size.height()

        if selected:
            qpen = QPen(QColor(238, 136, 68))
        else:
            qpen = QPen(QColor(200, 200, 200))
        qpen.setWidth(2)
        qp.setPen(qpen)

        qp.setBrush(QBrush(c))
        qp.drawRect(int(f * w), 0, self.handler_w, h-1)
    
    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        size = self.size()
        w = size.width() - self.handler_w
        for i in range(len(self.color_ramps)):
            f = self.color_ramps[i][0]
            if abs(f * w + self.handler_w / 2 - e.x()) < self.handler_w / 2:
                self.select_index = i
                self.pressed = True
                f, (r, g, b) = self.color_ramps[i]
                self.updatePanel[QColor].emit(QColor(int(r * 255), int(g * 255), int(b * 255)))
        self.update()

    def mouseReleaseEvent(self, e):
        super().mousePressEvent(e)
        self.pressed = False
    
    def mouseMoveEvent(self, event):
        size = self.size()
        w = size.width()
        x = event.x()
        if self.pressed:
            ratio = color_picker_clamp(x / w, 0, 1)
            if self.select_index != 0:
                p = self.color_ramps[self.select_index - 1]
                ratio = max(p[0], ratio)
            if self.select_index != len(self.color_ramps) - 1:
                n = self.color_ramps[self.select_index + 1]
                ratio = min(n[0], ratio)
            self.color_ramps[self.select_index][0] = ratio
            self.update()
    
    def change_current_bar_color(self, c):
        item = self.color_ramps.pop(self.select_index)
        item[1] = (c.redF(), c.greenF(), c.blueF())
        self.color_ramps.insert(self.select_index, item)
        self.update()
        

class ColorRampBarWidget(QWidget):
    
    def __init__(self, parent, color_ramps):
        super().__init__()
        self.p = parent
        self.color_ramps = color_ramps
        
        self.initUI()
        
        
    def initUI(self):
        add_btn = QPushButton(self)
        add_btn.setText('+')
        add_btn.setFixedWidth(30)

        del_btn = QPushButton(self)
        del_btn.setText('-')
        del_btn.setFixedWidth(30)

        preset_combo = QComboBox(self)
        preset_combo.addItem('BlackBody')
        preset_combo.addItem('Grayscale')
        preset_combo.addItem('InfraRed')
        preset_combo.addItem('TwoTone')
        preset_combo.addItem('WhiteToRed')
        preset_combo.setFixedWidth(100)

        self.wid = BurningWidget(self.p.updateBar, self.p.updatePanel, self.color_ramps)
        self.wid.setFixedHeight(20)
        add_btn.clicked.connect(self.wid.add)
        del_btn.clicked.connect(self.wid.remove)
        preset_combo.textActivated.connect(self.wid.set_preset)
        
        hbox = QHBoxLayout()
        hbox.addWidget(add_btn)
        hbox.addWidget(del_btn)
        hbox.addStretch(1)
        hbox.addWidget(preset_combo)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.wid)
        self.setLayout(vbox)

        self.setGeometry(300, 300, 390, 210)
        self.setWindowTitle('Burning widget')
        self.show()


class ColorPickerWindow(QWidget):
    updatePanel = Signal(QColor)
    updateBar = Signal(QColor)

    def __init__(self, color_ramps):
        super().__init__()
        self.color_ramps = color_ramps
        self.initUI()

    def initUI(self):

        self.colorRampBarWidget = ColorRampBarWidget(self, self.color_ramps)
        self.colorWall = ColorWall(self)
        self.colorPickerWidget = ColorPickerWidget(self)

        vbox = QVBoxLayout()
        vbox.addWidget(self.colorRampBarWidget)
        vbox.addWidget(self.colorWall)
        vbox.addWidget(self.colorPickerWidget)

        self.setLayout(vbox)
        self.setGeometry(300, 300, 390, 210)
        self.setWindowTitle('Color picker')
        self.show()
        
        f, (r, g, b) = self.color_ramps[0]
        self.updatePanel[QColor].emit(QColor(int(r * 255), int(g * 255), int(b * 255)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    colorPickerDialog = ColorPickerWindow(color_ramps)
    colorPickerDialog.show()
    app.exec_()
