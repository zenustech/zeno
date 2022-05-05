import json

from .dll import core
from .serial import serializeScene
from PySide2.QtCore import QSettings

def evaluateExpr(expr, frame):
    try:
        return eval('f' + repr(expr))
    except:
        return expr


def runScene(graphs, nframes, iopath, start_frame):
    core.setIOPath(iopath)

    data = json.dumps(list(serializeScene(graphs)))
    core.loadScene(data)
    #for cmd, *args in serializeScene(graphs):
    #    getattr(core, cmd)(*args)

    applies = set()
    nodes = graphs['main']['nodes']
    for ident, data in nodes.items():
        if 'special' in data:
            continue
        options = data['options']
        if 'VIEW' in options:
            applies.add(ident)

    core.switchGraph('main')
    core.setFrameid(start_frame)

    for frameid in range(start_frame, start_frame + nframes):
        print('FRAME:', frameid)
        ### BEGIN XINXIN HAPPY >>>>>
        setting = QSettings('ZenusTech','Zeno')
        nas_loc = setting.value('nas_loc')
        for ident, data in graphs['main']['nodes'].items():
            if 'special' in data:
                continue
            name = data['name']
            inputs = data['inputs']
            for name, value in inputs.items():
                if type(value) is list and len(value) == 3 and type(value[2]) == str:
                    value = value[2].replace('$NASLOC', nas_loc)
                    value = evaluateExpr(value, frameid)
                    core.setNodeInputString(ident, name, value)
            params = data['params']
            for name, value in params.items():
                if type(value) is str:
                    value = value.replace('$NASLOC', nas_loc)
                    value = evaluateExpr(value, frameid)
                    core.setNodeParam(ident, name, value)
        ### ENDOF XINXIN HAPPY <<<<<

        core.frameBegin()
        while core.substepBegin():
            core.applyNodes(applies)
            core.substepEnd()
        core.frameEnd()

    print('EXITING')


def dumpDescriptors():
    return core.dumpDescriptors()


__all__ = [
    'runScene',
    'dumpDescriptors',
]
