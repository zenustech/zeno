from .editor import *

class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width = style['node_width']
        self.height = 0

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        self.title.setPos(HORI_MARGIN * 2, -TEXT_HEIGHT)
        font = QFont()
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.collapsed = False
        self.collapse_button = QDMGraphicsCollapseButton(self)
        self.collapse_button.resize(28, 28)

        self.params = {}
        self.inputs = {}
        self.outputs = {}
        self.options = {}
        self.name = None
        self.ident = None

        self.desc_inputs = []
        self.desc_outputs = []
        self.desc_params = []

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.scene().moved = True
        for socket in self.inputs.values():
            for edge in socket.edges:
                edge.updatePath()
        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

    def remove(self):
        for socket in list(self.inputs.values()):
            socket.remove()
        for socket in list(self.outputs.values()):
            socket.remove()
        
        self.scene().nodes.remove(self)
        self.scene().removeItem(self)

    def setIdent(self, ident):
        self.ident = ident

    def setName(self, name):
        if self.ident is None:
            self.ident = gen_unique_ident(name)
        self.name = name
        self.title.setPlainText(name)

    def getOptions(self):
        collapsed_status = ['collapsed'] if self.collapsed else []
        return [name for name, button in self.options.items() if button.checked] + collapsed_status

    def setOptions(self, options):
        if 'collapsed' in options:
            self.collapse_button.setCollapsed(True)
        for name, button in self.options.items():
            button.setChecked(name in options)

    def initDummySockets(self):
        h = TEXT_HEIGHT / 2
        offset = style['dummy_socket_offset']
        s = QDMGraphicsSocket(self)
        s.setPos(-offset, h)
        s.setIsOutput(False)
        s.dummy = True
        self.dummy_input_socket = s
        self.dummy_input_socket.hide()

        w = 240
        s = QDMGraphicsSocket(self)
        s.setPos(w + offset, h)
        s.setIsOutput(False)
        s.dummy = True
        self.dummy_output_socket = s
        self.dummy_output_socket.hide()

    def initCondButtons(self):
        cond_keys = ['OUT', 'MUTE', 'ONCE', 'VIEW']
        for i, key in enumerate(cond_keys):
            button = QDMGraphicsButton(self)
            M = HORI_MARGIN // 2
            W = 38
            rect = QRect(W * i + M, -38, 34, 34)
            button.setGeometry(rect)
            button.setText(key)
            self.options[key] = button

    def initSockets(self):
        self.initDummySockets()
        self.initCondButtons()

        inputs = self.desc_inputs
        outputs = self.desc_outputs
        params = self.desc_params

        y = TEXT_HEIGHT * 0.4

        self.params.clear()
        for index, (type, name, defl) in enumerate(params):
            param = globals()['QDMGraphicsParam_' + type](self)
            rect = QRect(HORI_MARGIN, y, self.width - HORI_MARGIN * 2, 0)
            param.setGeometry(rect)
            param.setName(name)
            param.setDefault(defl)
            self.params[name] = param
            y += param.geometry().height()

        if len(params):
            y += TEXT_HEIGHT * 0.7
        else:
            y += TEXT_HEIGHT * 0.4

        socket_start = y + TEXT_HEIGHT * style['output_shift']

        self.inputs.clear()
        for index, name in enumerate(inputs):
            socket = QDMGraphicsSocket(self)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(False)
            self.inputs[name] = socket
            y += TEXT_HEIGHT

        y = socket_start
        if len(inputs) > len(outputs):
            y += (len(inputs) - len(outputs)) * TEXT_HEIGHT

        self.outputs.clear()
        for index, name in enumerate(outputs):
            socket = QDMGraphicsSocket(self)
            index += len(params) + len(inputs)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(True)
            self.outputs[name] = socket
            y += TEXT_HEIGHT

        y = socket_start + max(len(inputs), len(outputs)) * TEXT_HEIGHT

        self.height = y

        self.title.setPos(HORI_MARGIN * 2, self.height)
        self.collapse_button.setPos(204 + 4, self.height + 4)

    def boundingRect(self):
        top = 42
        bottom = 36

        h = 0 if self.collapsed else self.height
        h += (top + bottom)
        return QRectF(0, -top, self.width, h).normalized()

    def paint(self, painter, styleOptions, widget=None):
        top = 42
        bottom = 36

        if self.isSelected():
            pad = 10
            h = 0 if self.collapsed else self.height
            rect = QRect(-pad, -pad -top, self.width + pad * 2, h + pad * 2 + top + bottom)
            fillRect(painter, rect, '#52331F', 2, '#FA6400')

        r = style['node_rounded_radius']

        title_outline_color = '#787878'

        w = style['line_width']
        hw = w / 2

        line_width = style['line_width']
        line_color = title_outline_color

        y = 0 if self.collapsed else self.height
        # title background
        rect = QRect(0 + hw, y + hw, 203 - w, 36 - w)
        fillRect(painter, rect, style['title_color'], line_width, line_color)

        # collpase button background
        rect = QRect(204 + hw, y + hw, 36 - w, 36 - w)
        fillRect(painter, rect, style['title_color'], line_width, line_color)

        # button background
        rect = QRect(0, -top, 240, top)
        fillRect(painter, rect, '#638E77')

        M = HORI_MARGIN // 2
        W = 38
        rect = QRect(W * 4 + M, -38, 79, 34)
        fillRect(painter, rect, '#376557')

        # content panel background
        if not self.collapsed:
            rect = QRect(hw + w, hw, self.width - w * 3, self.height - w)
            fillRect(painter, rect, style['panel_color'], line_width, '#4a4a4a')


    def collapse(self):
        self.dummy_input_socket.show()
        self.dummy_output_socket.show()

        for v in self.params.values():
            v.hide()
        for v in self.inputs.values():
            v.hide()
        for v in self.outputs.values():
            v.hide()

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

        self.title.setPos(HORI_MARGIN * 2, 0)
        self.collapse_button.setPos(204 + 4, 4)

    def unfold(self):
        self.dummy_input_socket.hide()
        self.dummy_output_socket.hide()

        for v in self.params.values():
            v.show()
        for v in self.inputs.values():
            v.show()
        for v in self.outputs.values():
            v.show()

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

        self.title.setPos(HORI_MARGIN * 2, self.height)
        self.collapse_button.setPos(204 + 4, self.height + 4)

    def dump(self):
        node = self
        inputs = {}
        for name, socket in node.inputs.items():
            assert not socket.isOutput
            data = None
            if socket.hasAnyEdge():
                srcSocket = socket.getTheOnlyEdge().srcSocket
                data = srcSocket.node.ident, srcSocket.name
            inputs[name] = data

        params = {}
        for name, param in node.params.items():
            value = param.getValue()
            params[name] = value

        uipos = node.pos().x(), node.pos().y()
        options = node.getOptions()

        data = {
            'name': node.name,
            'inputs': inputs,
            'params': params,
            'uipos': uipos,
            'options': options,
        }
        return {node.ident: data}
    
    def load(self, ident, data):
        node = self
        name = data['name']
        inputs = data['inputs']
        params = data['params']
        posx, posy = data['uipos']
        options = data.get('options', [])

        node.initSockets()
        node.setIdent(ident)
        node.setName(name)
        node.setPos(posx, posy)
        node.setOptions(options)

        for name, value in params.items():
            if name not in node.params:
                print('no param named [{}] for [{}]'.format(
                    name, nodes[ident]['name']))
                continue
            param = node.params[name]
            param.setValue(value)

        edges = []
        for name, input in inputs.items():
            if input is None:
                continue
            if name not in node.inputs:
                print('no input named [{}] for [{}]'.format(
                    name, nodes[ident]['name']))
                continue
            dest = node.inputs[name]
            edges.append((dest, input))
        return edges

