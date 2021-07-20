from .editor import *

style = {
    'title_color': '#638e77',
    'socket_connect_color': '#638e77',
    'socket_unconnect_color': '#4a4a4a',
    'title_text_color': '#FFFFFF',
    'title_text_size': 10,
    'button_text_size': 10,
    'socket_text_size': 10,
    'param_text_size': 10,
    'socket_text_color': '#FFFFFF',
    'panel_color': '#282828',
    'blackboard_title_color': '#393939',
    'blackboard_panel_color': '#1B1B1B',
    'line_color': '#B0B0B0',
    'background_color': '#2C2C2C',
    'selected_color': '#EE8844',
    'button_color': '#1e1e1e',
    'button_text_color': '#ffffff',
    'button_selected_color': '#449922',
    'button_selected_text_color': '#333333',
    'output_shift': 1,

    'line_width': 3,
    'node_outline_width': 2,
    'socket_outline_width': 2,
    'node_rounded_radius': 6,
    'socket_radius': 8,
    'node_width': 200,
    'text_height': 23,
    'copy_offset_x': 100,
    'copy_offset_y': 100,
    'hori_margin': 9,
    'dummy_socket_offset': 15,
}


class HistoryStack:
    def __init__(self, scene):
        self.scene = scene
        self.init_state()
    
    def init_state(self):
        self.current_pointer = -1
        self.stack = []

    def undo(self):
        # can not undo at stack bottom
        if self.current_pointer == 0:
            return

        self.current_pointer -= 1
        current_scene = self.stack[self.current_pointer]
        self.scene.newGraph()
        self.scene.loadGraph(current_scene)

    def redo(self):
        # can not redo at stack top
        if self.current_pointer == len(self.stack) - 1:
            return

        self.current_pointer += 1
        current_scene = self.stack[self.current_pointer]
        self.scene.newGraph()
        self.scene.loadGraph(current_scene)
    
    def record(self):
        if self.current_pointer != len(self.stack) - 1:
            self.stack = self.stack[:self.current_pointer + 1]

        nodes = self.scene.dumpGraph()
        self.stack.append(nodes)
        self.current_pointer += 1

        # limit the stack length
        if self.current_pointer > MAX_STACK_LENGTH:
            self.stack = self.stack[1:]
            self.current_pointer = MAX_STACK_LENGTH


class QDMSearchLineEdit(QLineEdit):
    def __init__(self, menu, view):
        super().__init__(menu)
        self.menu = menu
        self.view = view
        self.wact = QWidgetAction(self.menu)
        self.wact.setDefaultWidget(self)
        self.menu.addAction(self.wact)


class QDMFileMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('&File')

        acts = [
                ('&New', QKeySequence.New),
                ('&Open', QKeySequence.Open),
                ('&Save', QKeySequence.Save),
                ('&Import', 'ctrl+shift+o'),
                ('Save &as', QKeySequence.SaveAs),
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
                ('Undo', QKeySequence.Undo),
                ('Redo', QKeySequence.Redo),
                (None, None),
                ('Copy', QKeySequence.Copy),
                ('Paste', QKeySequence.Paste),
        ]
        
        for name, shortcut in acts:
            if not name:
                self.addSeparator()
                continue
            action = QAction(name, self)
            action.setShortcut(shortcut)
            self.addAction(action)
