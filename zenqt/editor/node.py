from . import *


class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        font = QFont()
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
        self.initTopButtons()

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

        s = QDMGraphicsDummySocket(self)
        s.setPos(0, 0)
        s.setIsOutput(True)
        self.dummy_output_socket = s
        self.dummy_output_socket.hide()

    def initTopButtons(self):
        cond_keys = ['ONCE', 'PREP', 'MUTE', 'VIEW']
        for i, key in enumerate(cond_keys):
            button = QDMGraphicsTopButton(self)
            m = HORI_MARGIN // 2
            s = style['button_svg_size']
            offset_x = style['button_svg_offset_x']
            offset_y = style['button_svg_offset_y']
            rect = QRect(offset_x * i + m, -offset_y, s, s)
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

        y = self.height + TEXT_HEIGHT * 0.4

        self.params.clear()
        for index, (type, name, defl) in enumerate(params):
            param = globals()['QDMGraphicsParam_' + type](self)
            rect = QRectF(HORI_MARGIN, y, self.width - HORI_MARGIN * 2, 0)
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
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(True)
            self.outputs[name] = socket
            y += TEXT_HEIGHT

        y = socket_start + max(len(inputs), len(outputs)) * TEXT_HEIGHT

        y += TEXT_HEIGHT * 0.75
        self.height = y

        self.title.setPos(HORI_MARGIN * 0.8, self.height)
        pad = style['button_svg_padding']
        self.collapse_button.setPos(style['node_title_width'] + pad, self.height + pad)

    def boundingRect(self):
        h = TEXT_HEIGHT if self.collapsed else self.height
        return QRectF(0, -TEXT_HEIGHT, self.width, h).normalized()

    def paint(self, painter, styleOptions, widget=None):
        gap = style['gap']

        w = style['line_width']
        hw = w / 2

        line_width = style['line_width']
        line_color = style['title_outline_color']

        y = 0 if self.collapsed else self.height
        # title background
        rect = QRect(0, y + gap, style['node_title_width'] - gap, style['node_title_height'])
        fillRect(painter, rect, style['title_color'], line_width, line_color)

        # collpase button background
        rect = QRect(style['node_title_width'], y + gap,
                    style['node_collpase_width'], style['node_title_height'])
        fillRect(painter, rect, style['title_color'], line_width, line_color)

        # button background
        top = style['node_top_bat_height']
        node_width = style['node_width']
        rect = QRect(0, -top, node_width, top)
        fillRect(painter, rect, style['title_color'])

        M = HORI_MARGIN // 2
        s = style['button_svg_size']
        offset_x = style['button_svg_offset_x']
        offset_y = style['button_svg_offset_y']
        x = offset_x * 4 + M
        rect_w = node_width - x - M
        rect = QRect(x, -offset_y, rect_w, s)
        fillRect(painter, rect, style['top_button_color'])

        # content panel background
        if not self.collapsed:
            rect = QRect(hw + w, hw, self.width - w * 3, self.height - gap)
            fillRect(painter, rect, style['panel_color'], line_width, style['panel_outline_color'])

    def collapse(self):
        self.dummy_input_socket.show()
        self.dummy_output_socket.show()

        self.collapsed = True
        self.collapse_button.update_svg(self.collapsed)
        for v in self.options.values():
            v.hide()
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
        pad = style['button_svg_padding']
        self.collapse_button.setPos(style['node_title_width'] + pad, pad)

    def unfold(self):
        self.dummy_input_socket.hide()
        self.dummy_output_socket.hide()

        self.collapsed = False
        self.collapse_button.update_svg(self.collapsed)
        for v in self.options.values():
            v.show()
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
        pad = style['button_svg_padding']
        self.collapse_button.setPos(style['node_title_width'] + pad, self.height + pad)

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

