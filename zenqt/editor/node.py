from . import *


class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        font = QFont()
        font.setWeight(QFont.DemiBold)
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.collapse_button = QDMGraphicsCollapseButton(self)
        self.collapsed = False
        self.name = None
        self.ident = None
        self.desc = {'inputs': [], 'outputs': [], 'params': []}

        self.width = style['node_width']
        self.height = 0
        self.params = {}
        self.inputs = {}
        self.outputs = {}
        self.options = {}

        self.initDummySockets()
        self.initCondButtons()

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
            self.collapse()
        for name, button in self.options.items():
            button.setChecked(name in options)

    def initDummySockets(self):
        s = QDMGraphicsDummySocket(self)
        s.setPos(0, 0)
        s.setIsOutput(False)
        self.dummy_input_socket = s
        self.dummy_input_socket.hide()

        w = style['node_width']
        s = QDMGraphicsDummySocket(self)
        s.setPos(0, 0)
        s.setIsOutput(True)
        self.dummy_output_socket = s
        self.dummy_output_socket.hide()

    def initCondButtons(self):
        cond_keys = ['ONCE', 'PREP', 'MUTE', 'VIEW']
        for i, key in enumerate(cond_keys):
            button = QDMGraphicsTopButton(self)
            M = HORI_MARGIN // 2
            W = 38
            R = style['button_svg_size']
            H = style['button_svg_offset_y']
            rect = QRect(W * i + M, -H, R, R)
            button.setGeometry(rect)
            button.setText(key)
            self.options[key] = button

    def resetSockets(self):
        self.height = 0
        for param in list(self.params.values()):
            self.scene().removeItem(param)
        for input in list(self.inputs.values()):
            input.remove()
            self.scene().removeItem(input)
        for output in list(self.outputs.values()):
            output.remove()
            self.scene().removeItem(output)

    def initSockets(self):
        inputs = self.desc['inputs']
        outputs = self.desc['outputs']
        params = self.desc['params']

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

        socket_start = y
        socket_offset = 24

        self.inputs.clear()
        for index, name in enumerate(inputs):
            socket = QDMGraphicsSocket(self)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(False)
            self.inputs[name] = socket
            y += socket_offset

        y = socket_start
        if len(inputs) > len(outputs):
            y += (len(inputs) - len(outputs)) * socket_offset

        self.outputs.clear()
        for index, name in enumerate(outputs):
            socket = QDMGraphicsSocket(self)
            index += len(params) + len(inputs)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(True)
            self.outputs[name] = socket
            y += socket_offset

        y = socket_start + max(len(inputs), len(outputs)) * TEXT_HEIGHT

        self.height = y

        self.title.setPos(HORI_MARGIN * 0.8, self.height)
        self.collapse_button.setPos(204 + 6, self.height + 4)

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
            fillRectOld(painter, rect, QColor(82,51,31, 150), 2, '#FA6400')

        r = style['node_rounded_radius']

        title_outline_color = '#787878'

        w = style['line_width']
        hw = w / 2

        line_width = style['line_width']
        line_color = title_outline_color

        y = 0 if self.collapsed else self.height
        # title background
        rect = QRect(0, y + hw, 206, 36 - w)
        fillRect(painter, rect, style['title_color'], line_width, line_color)

        # collpase button background
        rect = QRect(206 + hw, y + hw, 36 - w, 36 - w)
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
            rect = QRect(hw + w, hw, self.width - w * 3, self.height - 1)
            fillRect(painter, rect, style['panel_color'], line_width, '#4a4a4a')

    def collapse(self):
        self.dummy_input_socket.show()
        self.dummy_output_socket.show()

        self.collapsed = True
        self.collapse_button.update_svg(self.collapsed)
        for v in self.params.values():
            v.hide()
        for v in self.inputs.values():
            v.hide()
        for v in self.outputs.values():
            v.hide()

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

        self.title.setPos(HORI_MARGIN * 0.8, 0)
        self.collapse_button.setPos(204 + 6, 4)

    def unfold(self):
        self.dummy_input_socket.hide()
        self.dummy_output_socket.hide()

        self.collapsed = False
        self.collapse_button.update_svg(self.collapsed)
        for v in self.params.values():
            v.show()
        for v in self.inputs.values():
            v.show()
        for v in self.outputs.values():
            v.show()

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

        self.title.setPos(HORI_MARGIN * 0.8, self.height)
        self.collapse_button.setPos(204 + 6, self.height + 4)

    def dump(self):
        inputs = {}
        for name, socket in self.inputs.items():
            assert not socket.isOutput
            data = None
            if socket.hasAnyEdge():
                srcSocket = socket.getTheOnlyEdge().srcSocket
                data = srcSocket.node.ident, srcSocket.name
            inputs[name] = data

        params = {}
        for name, param in self.params.items():
            value = param.getValue()
            params[name] = value

        uipos = self.pos().x(), self.pos().y()
        options = self.getOptions()

        data = {
            'name': self.name,
            'inputs': inputs,
            'params': params,
            'uipos': uipos,
            'options': options,
        }
        return self.ident, data

    def saveEdges(self):
        inputs = {}
        for name, socket in self.inputs.items():
            res = []
            for e in socket.edges:
                res.append((e.srcSocket.node.ident, e.srcSocket.name))
            inputs[name] = res
        outputs = {}
        for name, socket in self.outputs.items():
            res = []
            for e in socket.edges:
                res.append((e.dstSocket.node.ident, e.dstSocket.name))
            outputs[name] = res
        return inputs, outputs

    def restoreEdges(self, saved):
        nodesLut = {}
        for node in self.scene().nodes:
            nodesLut[node.ident] = node

        inputs, outputs = saved
        for name, socket in self.inputs.items():
            if name in inputs:
                for sn, ss in inputs[name]:
                    if sn in nodesLut:
                        node = nodesLut[sn]
                        if ss in node.outputs:
                            source = node.outputs[ss]
                            self.scene().addEdge(source, socket)
        for name, socket in self.outputs.items():
            if name in outputs:
                for dn, ds in outputs[name]:
                    if dn in nodesLut:
                        node = nodesLut[dn]
                        if ds in node.inputs:
                            dest = node.inputs[ds]
                            self.scene().addEdge(socket, dest)

    def reloadSockets(self):
        edges = self.saveEdges()
        self.resetSockets()
        self.initSockets()
        self.restoreEdges(edges)
    
    def load(self, ident, data):
        name = data['name']
        inputs = data['inputs']
        params = data['params']
        posx, posy = data['uipos']
        options = data.get('options', [])

        self.initSockets()
        self.setIdent(ident)
        self.setName(name)
        self.setPos(posx, posy)
        self.setOptions(options)

        for name, value in params.items():
            if name not in self.params:
                if not name.startswith('_'):
                    print('no param named [{}] for [{}]'.format(
                        name, data['name']))
                continue
            param = self.params[name]
            param.setValue(value)

        edges = []
        for name, input in inputs.items():
            if input is None:
                continue
            if name not in self.inputs:
                print('no input named [{}] for [{}]'.format(
                    name, data['name']))
                continue
            dest = self.inputs[name]
            edges.append((dest, input))
        return edges

