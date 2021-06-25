from zenutils import rel2abs

def asset_path(name):
    return rel2abs(__file__, 'assets', name)

from .mainwindow import MainWindow
