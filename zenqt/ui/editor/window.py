from . import *


class QDMFileMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('&File')

        acts = [
                ('&New', QKeySequence.New),
                ('&Open', QKeySequence.Open),
                ('&Save', QKeySequence.Save),
                ('&Import', 'Ctrl+Shift+O'),
                ('Save &as', QKeySequence.SaveAs),
                ('&Export', 'Ctrl+Shift+E'),
        ]

        for name, shortcut in acts:
            if not name:
                self.addSeparator()
                continue
            action = QAction(name, self)
            action.setShortcut(shortcut)
            self.addAction(action)


class QDMEditMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('&Edit')

        acts = [
                ('&Undo', QKeySequence.Undo),
                ('&Redo', QKeySequence.Redo),
                (None, None),
                ('&Copy', QKeySequence.Copy),
                ('&Paste', QKeySequence.Paste),
                (None, None),
                ('&Find', QKeySequence.Find),
                ('Easy Subgraph', 'Alt+S'),
        ]
        
        for name, shortcut in acts:
            if not name:
                self.addSeparator()
                continue
            action = QAction(name, self)
            action.setShortcut(shortcut)
            self.addAction(action)

class SubgraphHistoryStack:
    def __init__(self, node_editor):
        self.node_editor = node_editor
        self.stack = ['main']
        self.pointer = 0

    def bind(self, undo_button, redo_button):
        self.undo_button = undo_button
        self.redo_button = redo_button

    def record(self, name):
        if self.pointer != len(self.stack) - 1:
            self.stack = self.stack[:self.pointer + 1]

        if self.stack[-1] != name:
            self.stack.append(name)
            self.pointer += 1

    def undo(self):
        if self.pointer == 0:
            return

        self.pointer -= 1
        name = self.stack[self.pointer]
        self.checkout(name)

    def redo(self):
        if self.pointer == len(self.stack) - 1:
            return

        self.pointer += 1
        name = self.stack[self.pointer]
        self.checkout(name)

    def checkout(self, name):
        if name in self.node_editor.scenes:
            self.node_editor.switchScene(name)
            self.node_editor.edit_graphname.setCurrentText(name)

class NodeEditor(QWidget):
    def __init__(self, parent, window):
        super().__init__(parent)

        self.window = window

        self.always_run = False
        self.target_frame = 0

        self.current_path = None
        self.clipboard = QApplication.clipboard()

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.menubar = QMenuBar()
        self.layout.addWidget(self.menubar)

        self.menuFile = QDMFileMenu()
        self.menuFile.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.menuFile)

        self.menuEdit = QDMEditMenu()
        self.menuEdit.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.menuEdit)

        self.view = QDMGraphicsView(self)
        self.layout.addWidget(self.view)

        self.scenes = {}
        self.descs = {}
        self.cates = {}
        self.descs_comment = {}

        self.initExecute()
        self.initShortcuts()
        self.initDescriptors()
        self.initDescriptorsComment()

        self.newProgram()

        self.startTimer(1000 * 10)

    def try_run_this_frame(self, frame=None):
        if frame != None:
            self.target_frame = frame
        if self.always_run:
            prog = self.dumpProgram()
            go(launch.launchProgram, prog, nframes=1, start_frame=self.target_frame)
            print('run_this_frame')

    @property
    def current_path(self):
        return self._current_path

    @current_path.setter
    def current_path(self, value):
        self._current_path = value
        self.window.setWindowTitleWithPostfix(value)

    def timerEvent(self, event):
        self.auto_save()
        super().timerEvent(event)

    def auto_save(self):
        if any(s.contentChanged is True for s in self.scenes.values()):
            from ...system.utils import os_name
            if os_name == 'win32':
                dir_path = '\\zeno_autosave'
            else:
                dir_path = '/tmp/autosave'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            file_name += '.zsg'
            path = os.path.join(dir_path, file_name)
            #print('auto saving to', path)
            self.do_save(path, auto_save=True)
            for s in self.scenes.values():
                if s.contentChanged:
                    # True for not autosaved, not manualsaved
                    # 'AUTOSAVED' for not manualsaved, autosaved
                    # False for manualsaved, autosaved
                    s.contentChanged = 'AUTOSAVED'

    def clearScenes(self):
        self.scenes.clear()

    def deleteCurrScene(self):
        for k, v in self.scenes.items():
            if v is self.scene:
                del self.scenes[k]
                break
        self.switchScene('main')

    def switchScene(self, name):
        if name not in self.scenes:
            scene = QDMGraphicsScene()
            scene.editor = self
            scene.record()
            scene.setContentChanged(False)
            scene.name = name
            self.scenes[name] = scene
        else:
            scene = self.scenes[name]
        self.view.setScene(scene)
        self.edit_graphname.clear()
        self.edit_graphname.addItems(self.scenes.keys())

    @property
    def scene(self):
        return self.view.scene()

    def handleEnvironParams(self):
        if os.environ.get('ZEN_OPEN'):
            path = os.environ['ZEN_OPEN']
            self.do_open(path)
            self.current_path = path

        if os.environ.get('ZEN_DOEXEC'):
            print('ZEN_DOEXEC found, direct execute')
            self.on_execute()

    def showEvent(self, event):
        super().showEvent(event)
        self.handleEnvironParams()

    def initShortcuts(self):
        self.msgDel = QShortcut(QKeySequence('Del'), self)
        self.msgDel.activated.connect(self.on_delete)

        self.msgAdd = QShortcut(QKeySequence('Tab'), self)
        self.msgAdd.activated.connect(self.shortcut_add)

    def initExecute(self):
        self.edit_graphname = QComboBox(self)
        self.edit_graphname.setEditable(True)
        self.edit_graphname.move(20, 40)
        self.edit_graphname.resize(130, 30)
        self.edit_graphname.textActivated.connect(self.on_switch_graph)

        self.button_new = QPushButton('New', self)
        self.button_new.move(160, 40)
        self.button_new.resize(80, 30)
        self.button_new.clicked.connect(self.on_new_graph)

        self.button_delete = QPushButton('Delete', self)
        self.button_delete.move(250, 40)
        self.button_delete.resize(80, 30)
        self.button_delete.clicked.connect(self.deleteCurrScene)

        self.button_undo = QPushButton('<-', self)
        self.button_undo.move(20, 80)
        self.button_undo.resize(60, 30)

        self.button_redo = QPushButton('->', self)
        self.button_redo.move(90, 80)
        self.button_redo.resize(60, 30)

        self.find_bar = QDMFindBar(self)
        self.find_bar.move(400, 40)
        self.find_bar.resize(320, 30)
        self.find_bar.hide()

        self.subgraphHistoryStack = SubgraphHistoryStack(self)
        self.subgraphHistoryStack.bind(self.button_undo, self.button_redo)
        self.button_undo.clicked.connect(self.subgraphHistoryStack.undo)
        self.button_redo.clicked.connect(self.subgraphHistoryStack.redo)

    def on_switch_graph(self, name):
        self.subgraphHistoryStack.record(name)
        self.switchScene(name)
        self.initDescriptors()
        self.scene.reloadNodes()
        self.edit_graphname.setCurrentText(name)
        self.view.check_scene_rect()

    def on_new_graph(self):
        name = self.edit_graphname.currentText()
        self.on_switch_graph(name)

    def setDescriptors(self, descs):
        self.descs = descs
        self.cates.clear()
        for name, desc in self.descs.items():
            for cate in desc['categories']:
                self.cates.setdefault(cate, []).append(name)

    def initDescriptors(self):
        descs = launch.getDescriptors()
        subg_descs = self.getSubgraphDescs()
        descs.update(subg_descs)
        descs.update({
            'Blackboard': {
                'inputs': [],
                'outputs': [],
                'params': [],
                'categories': ['layout'],
            },
        })
        self.setDescriptors(descs)

    def initDescriptorsComment(self):
        with open(asset_path('descs_comment.json'), 'r') as f:
            self.descs_comment = json.load(f)

    def on_add(self):
        pos = QPointF(0, 0)
        self.view.contextMenu(pos)

    def shortcut_add(self):
        pos = self.view.mapFromGlobal(QCursor.pos())
        self.view.contextMenu(pos)

    def on_kill(self):
        launch.killProcess()

    def dumpProgram(self):
        graphs = {}
        views = {}
        for name, scene in self.scenes.items():
            nodes = scene.dumpGraph()
            graphs[name] = {'nodes': nodes}
            if scene._scene_rect:
                graphs[name]['view_rect'] = {
                    'x': scene._scene_rect.x(),
                    'y': scene._scene_rect.y(),
                    'width': scene._scene_rect.width(),
                    'height': scene._scene_rect.height(),
                }

        prog = {}
        prog['graph'] = graphs
        prog['views'] = views
        prog['descs'] = dict(self.descs)
        prog['version'] = CURR_VERSION
        prog['viewport'] = {
            'camera_record': zenvis.status['camera_keyframes'],
            'lights': zenvis.dump_lights(),
        }
        return prog

    def bkwdCompatProgram(self, prog):
        if 'graph' not in prog:
            if 'main' not in prog:
                prog = {'main': prog}
            prog = {'graph': prog}
        if 'descs' not in prog:
            prog['descs'] = dict(self.descs)

        for name, desc in prog['descs'].items():
            for key, output in enumerate(desc['outputs']):
                if isinstance(output, str):
                    desc['outputs'][key] = ('', output, '')
            for key, input in enumerate(desc['inputs']):
                if isinstance(input, str):
                    desc['inputs'][key] = ('', input, '')

        for name, graph in prog['graph'].items():
            if 'nodes' not in graph:
                prog['graph'][name] = {
                    'nodes': graph,
                    'view_rect': {
                        'scale': 1,
                        'x': 0,
                        'y': 0,
                    },
                }

        for name, graph in prog['graph'].items():
            if 'view' in graph:
                graph['view_rect'] = {
                    'x': graph['view']['trans_x'],
                    'y': graph['view']['trans_y'],
                    'width': 1200 / graph['view']['scale'],
                    'height': 1000 / graph['view']['scale'],
                }

        if 'version' not in prog:
            prog['version'] = 'v0'
        if prog['version'] != CURR_VERSION:
            print('WARNING: Loading graph of version', prog['version'],
                'with editor version', CURR_VERSION)
        return prog

    def importProgram(self, prog):
        prog = self.bkwdCompatProgram(prog)
        self.setDescriptors(prog['descs'])
        # self.scene.newGraph()
        for name, graph in prog['graph'].items():
            if name != 'main':
                print('Importing subgraph', name)
                self.switchScene(name)
                nodes = graph['nodes']
                self.scene.loadGraphEx(graph)
                self.scene.history_stack.init_state()
                self.scene.record()
        self.scene.record()
        self.switchScene('main')
        self.initDescriptors()

    def loadProgram(self, prog):
        prog = self.bkwdCompatProgram(prog)
        self.setDescriptors(prog['descs'])
        self.clearScenes()
        for name, graph in prog['graph'].items():
            print('Loading subgraph', name)
            self.switchScene(name)
            self.scene.loadGraphEx(graph)
            self.scene.history_stack.init_state()
            self.scene.record()
        self.initDescriptors()
        self.switchScene('main')
        if 'viewport' in prog:
            s = prog['viewport']
            for k, v in s['camera_record'].items():
                zenvis.status['camera_keyframes'][int(k)] = v
            if 'lights' in s:
                zenvis.load_lights(s['lights'])

    def on_execute(self):
        nframes = int(self.edit_nframes.text())
        prog = self.dumpProgram()
        go(launch.launchProgram, prog, nframes, start_frame=0)

    def on_delete(self):
        itemList = self.scene.selectedItems()
        if not itemList: return
        for item in itemList:
            if item.scene() is not None:
                item.remove()
        self.scene.record()

    def newProgram(self):
        self.clearScenes()
        self.switchScene('main')

    def getOpenFileName(self):
        path, kind = QFileDialog.getOpenFileName(self, 'File to Open',
                '', 'Zensim Graph File(*.zsg);; All Files(*);;')
        return path

    def getSaveFileName(self):
        path, kind = QFileDialog.getSaveFileName(self, 'Path to Save',
                '', 'Zensim Graph File(*.zsg);; All Files(*);;')
        return path

    def menuTriggered(self, act):
        name = act.text()
        if name == '&New':
            if not self.confirm_discard('New'):
                return
            self.newProgram()
            self.current_path = None

        elif name == '&Open':
            if not self.confirm_discard('Open'):
                return
            path = self.getOpenFileName()
            if path:
                self.do_open(path)
                self.current_path = path

        elif name == 'Save &as' or (name == '&Save' and self.current_path is None):
            path = self.getSaveFileName()
            if path:
                self.do_save(path)
                self.current_path = path

        elif name == '&Save':
            self.do_save(self.current_path)

        elif name == '&Import':
            path = self.getOpenFileName()
            if path != '':
                self.do_import(path)

        elif name == '&Export':
            self.do_export()

        elif name == '&Undo':
            self.scene.undo()

        elif name == '&Redo':
            self.scene.redo()

        elif name == '&Copy':
            self.do_copy()

        elif name == '&Paste':
            self.do_paste()

        elif name == '&Find':
            self.find_bar.show()

        elif name == 'Easy Subgraph':
            self.easy_subgraph()

    def do_export(self):
        path, kind = QFileDialog.getSaveFileName(self, 'Path to Export',
                '', 'C++ Source File(*.cpp);; C++ Header File(*.h);; JSON file(*.json);; All Files(*);;',
                options=QFileDialog.DontConfirmOverwrite)
        if path != '':
            prog = self.dumpProgram()
            from ...system import serial

            if path.endswith('.cpp'):
                graphs = serial.serializeGraphs(prog['graph'], has_subgraphs=False)
                content = self.do_export_cpp(graphs)
            else:
                data = list(serial.serializeScene(prog['graph']))
                content = json.dumps(data)
                if path.endswith('.h'):
                    content = 'R"ZSL(' + content + ')ZSL"\n'

            with open(path, 'w') as f:
                f.write(content)

    def do_export_cpp(self, graphs):
        res = '/* auto generated from: %s */\n' % self.current_path
        res += '#include <zeno/zeno.h>\n'
        res += '#include <zeno/extra/ISubgraphNode.h>\n'
        res += 'namespace {\n'

        for key, data in graphs.items():
            if key not in self.descs: continue
            desc = self.descs[key]
            res += 'struct ' + key + ''' : zeno::ISerialSubgraphNode {
    virtual const char *get_subgraph_json() override {
        return R"ZSL(
''' + json.dumps(data) + '''
)ZSL";
    }
};
ZENDEFNODE(''' + key + ''', {
    {''' + ', '.join('{"%s", "%s", "%s"}' % (x, y, z) for x, y, z in desc['inputs'] if y != 'SRC') + '''},
    {''' + ', '.join('{"%s", "%s", "%s"}' % (x, y, z) for x, y, z in desc['outputs'] if y != 'DST') + '''},
    {''' + ', '.join('{"%s", "%s", "%s"}' % (x, y, z) for x, y, z in desc['params']) + '''},
    {''' + ', '.join('"%s"' % x for x in desc['categories']) + '''},
});
'''

        res += '}\n'
        return res

    def do_copy(self):
        itemList = self.scene.selectedItems()
        itemList = [n for n in itemList if isinstance(n, QDMGraphicsNode) or isinstance(n, QDMGraphicsNode_Blackboard) ]
        nodes = self.scene.dumpGraph(itemList)
        self.clipboard.setText(json.dumps(nodes))

    def do_paste(self):
            if self.clipboard.text() == '':
                print('nothing to paste')
                return
            itemList = self.scene.selectedItems()
            for i in itemList:
                i.setSelected(False)
            nodes = json.loads(self.clipboard.text())
            nid_map = {}
            for nid in nodes:
                nid_map[nid] = gen_unique_ident(nodes[nid]['name'])
            new_nodes = {}
            pos = self.view.mapToScene(self.view.mapFromGlobal(QCursor.pos()))
            coors = [n['uipos'] for n in nodes.values()]
            min_x = min(x for x, y in coors)
            min_y = min(y for x, y in coors)
            max_x = max(x for x, y in coors)
            max_y = max(y for x, y in coors)
            offset_x = pos.x() - (min_x + max_x) / 2
            offset_y = pos.y() - (min_y + max_y) / 2
            for nid, n in nodes.items():
                x, y = n['uipos']
                n['uipos'] = (x + offset_x, y + offset_y)
                if 'inputs' in n:
                    inputs = n['inputs']
                    for name, info in inputs.items():
                        if info == None:
                            continue
                        nid_, name_, value = info
                        if nid_ in nid_map and value != None:
                            info = (nid_map[nid_], name_, value)
                        elif nid_ in nid_map:
                            info = (nid_map[nid_], name_)
                        elif value != None:
                            info = (None, None, value)
                        else:
                            info = None
                        inputs[name] = info
                new_nodes[nid_map[nid]] = n
            self.scene.loadGraph(new_nodes, select_all=True)
            self.scene.record()

    def easy_subgraph(self):
        self.do_copy()
        name, okPressed = QInputDialog.getText(self, "New Subgraph", "Subgraph Name:", QLineEdit.Normal, "")
        if okPressed == False:
            return
        self.on_switch_graph(name)
        self.do_paste()

    def do_save(self, path, auto_save=False):
        prog = self.dumpProgram()
        with open(path, 'w') as f:
            json.dump(prog, f, indent=1)
        if not auto_save:
            for scene in self.scenes.values():
                scene.setContentChanged(False)

    def do_import(self, path):
        with open(path, 'r') as f:
            prog = json.load(f)
        self.importProgram(prog)

    def do_open(self, path):
        with open(path, 'r') as f:
            prog = json.load(f)
        self.loadProgram(prog)

    def confirm_discard(self, title):
        if os.environ.get('ZEN_OPEN'):
            return True
        contentChanged = False
        for scene in self.scenes.values():
            if scene.contentChanged:
                contentChanged = True
        if contentChanged:
            flag = QMessageBox.question(self, title, 'Discard unsaved changes?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return flag == QMessageBox.Yes
        return True

    def getSubgraphDescs(self):
        descs = {}
        for name, scene in self.scenes.items():
            if name == 'main': continue
            graph = scene.dumpGraph()
            subcategory = 'subgraph'
            subinputs = []
            suboutputs = []
            for node in graph.values():
                if node['name'] == 'SubInput':
                    params = node['params']
                    n_type = params.get('type')
                    n_name = params['name']
                    n_defl = params.get('defl')
                    subinputs.append((n_type, n_name, n_defl))
                elif node['name'] == 'SubOutput':
                    params = node['params']
                    n_type = params.get('type')
                    n_name = params['name']
                    n_defl = params.get('defl')
                    suboutputs.append((n_type, n_name, n_defl))
                elif node['name'] == 'SubCategory':
                    params = node['params']
                    subcategory = params['name']
            subinputs.extend(self.descs['Subgraph']['inputs'])
            suboutputs.extend(self.descs['Subgraph']['outputs'])
            desc = {}
            desc['inputs'] = subinputs
            desc['outputs'] = suboutputs
            desc['params'] = []
            desc['categories'] = [subcategory]
            desc['is_subgraph'] = True
            descs[name] = desc
        return descs

