import json

from .dll import core
from .serial import serializeScene


def evaluateExpr(expr, frame):
    try:
        return eval('f' + repr(expr))
    except:
        return expr


def runScene(graphs, nframes, iopath):
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

    for frameid in range(nframes):
        print('FRAME:', frameid)
        ### BEGIN XINXIN HAPPY >>>>>
        for ident, data in graphs['main']['nodes'].items():
            if 'special' in data:
                continue
            name = data['name']
            inputs = data['inputs']
            params = data['params']
            for name, value in params.items():
                if type(value) is str:
                    value = evaluateExpr(value, frameid)
                    rasrasr
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
