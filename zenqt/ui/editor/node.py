from . import *


class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        self.title.setPos(HORI_MARGIN * 2, -TEXT_HEIGHT)
        font = QFont()
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.collapse_button = QDMGraphicsCollapseButton(self)
        self.collapse_button.setPos(HORI_MARGIN * 0.5, -TEXT_HEIGHT * 0.84)
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
        self.title.setPlainText(translate(name))

    def getOptions(self):
        collapsed_status = ['collapsed'] if self.collapsed else []
        return [name for name, button in self.options.items() if button.checked] + collapsed_status

    def setOptions(self, options):
        if 'collapsed' in options:
            self.collapse()
        for name, button in self.options.items():
            button.setChecked(name in options)

    def initDummySockets(self):
        h = - TEXT_HEIGHT / 2
        offset = style['dummy_socket_offset']
        s = QDMGraphicsSocket(self)
        s.setPos(-offset, h)
        s.setIsOutput(False)
        s.dummy = True
        self.dummy_input_socket = s
        self.dummy_input_socket.hide()

        w = style['node_width']
        s = QDMGraphicsSocket(self)
        s.setPos(w + offset, h)
        s.setIsOutput(False)
        s.dummy = True
        self.dummy_output_socket = s
        self.dummy_output_socket.hide()

    def initCondButtons(self):
        cond_keys = ['ONCE', 'PREP', 'MUTE', 'VIEW']
        for i, key in enumerate(cond_keys):
            button = QDMGraphicsButton(self)
            M = HORI_MARGIN * 0.2
            H = TEXT_HEIGHT * 0.9
            W = self.width / len(cond_keys)
            button.setPos(W * i + M, -TEXT_HEIGHT * 2.3)
            button.setWidthHeight(W - M * 2, H)
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

        # todo: params are to be replaced by socket with default_value
        self.params.clear()
        for index, (type, name, defl) in enumerate(params):
            if type.startswith('enum '):
                param = QDMGraphicsParamEnum(self)
                enums = type.split()[1:]
                param.setEnums(enums)
                param.setZValue(len(params) - index)
            else:
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
        #if len(inputs) < len(outputs):
        #    y += (len(outputs) - len(inputs)) * TEXT_HEIGHT

        self.inputs.clear()
        for index, (type, name, defl) in enumerate(inputs):
            socket = QDMGraphicsSocket(self)
            socket.setIsOutput(False)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setType(type)
            socket.setDefault(defl)
            self.inputs[name] = socket
            y += TEXT_HEIGHT

        #y = socket_start
        #if len(inputs) > len(outputs):
        #    y += (len(inputs) - len(outputs)) * TEXT_HEIGHT

        self.outputs.clear()
        for index, (type, name, defl) in enumerate(outputs):
            socket = QDMGraphicsSocket(self)
            socket.setIsOutput(True)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setType(type)
            self.outputs[name] = socket
            y += TEXT_HEIGHT

        #y = socket_start + max(len(inputs), len(outputs)) * TEXT_HEIGHT

        y += TEXT_HEIGHT * 0.75
        self.height = y

    def boundingRect(self):
        h = TEXT_HEIGHT if self.collapsed else self.height
        return QRectF(0, -TEXT_HEIGHT, self.width, h).normalized()

    def paint(self, painter, styleOptions, widget=None):
        r = style['node_rounded_radius']

        if not self.collapsed:
            pathContent = QPainterPath()
            rect = QRectF(0, -TEXT_HEIGHT, self.width, self.height)
            pathContent.addRoundedRect(rect, r, r)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(style['panel_color']))
            painter.drawPath(pathContent.simplified())

            # title round top
            pathTitle = QPainterPath()
            rect = QRectF(0, -TEXT_HEIGHT, self.width, TEXT_HEIGHT)
            pathTitle.addRoundedRect(rect, r, r)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(style['title_color']))
            painter.drawPath(pathTitle.simplified())
            
            # title direct bottom
            pathTitle = QPainterPath()
            rect = QRectF(0, -r, self.width, r)
            pathTitle.addRect(rect)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(style['title_color']))
            painter.drawPath(pathTitle.simplified())

        pathOutline = QPainterPath()
        h = TEXT_HEIGHT if self.collapsed else self.height
        pathOutline.addRoundedRect(0, -TEXT_HEIGHT, self.width, h, r, r)
        pathOutlineColor = 'selected_color' if self.isSelected() else 'line_color'
        pen = QPen(QColor(style[pathOutlineColor]))
        pen.setWidth(style['node_outline_width'])
        painter.setPen(pen)
        if not self.collapsed:
            painter.setBrush(Qt.NoBrush)
        else:
            painter.setBrush(QColor(style['title_color']))
        painter.drawPath(pathOutline.simplified())

    def collapse(self):
        for v in self.childItems():
            if v not in [self.title, self.collapse_button]:
                v.hide()

        self.dummy_input_socket.show()
        self.dummy_output_socket.show()

        self.collapsed = True
        self.collapse_button.update_svg(self.collapsed)

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

    def unfold(self):
        for v in self.childItems():
            v.show()

        self.dummy_input_socket.hide()
        self.dummy_output_socket.hide()

        self.collapsed = False
        self.collapse_button.update_svg(self.collapsed)

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

    def dump(self):
        inputs = {}
        for name, socket in self.inputs.items():
            assert not socket.isOutput
            srcId = srcSock = None
            if socket.hasAnyEdge():
                srcSocket = socket.getTheOnlyEdge().srcSocket
                srcId, srcSock = srcSocket.node.ident, srcSocket.name
            deflVal = socket.getValue()
            inputs[name] = srcId, srcSock, deflVal

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
    
    def onInputChanged(self):
        pass

    def onOutputChanged(self):
        pass

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
            if input is None:  # bkwd-compat
                srcid = srcsock = None
            elif len(input) == 2:  # bkwd-compat
                srcid, srcsock = input
            else:
                srcid, srcsock, deflVal = input
                if deflVal is not None:
                    dest.setValue(deflVal)
            if srcid is not None:
                edges.append((dest, (srcid, srcsock)))
        return edges

