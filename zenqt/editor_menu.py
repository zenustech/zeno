from .editor import *

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


