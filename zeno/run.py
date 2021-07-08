import json

from .dll import core


def evaluateExpr(expr, frameid):
    frame = frameid
    return eval('f' + repr(expr))

def runScene(graphs, nframes, iopath):
    core.setIOPath(iopath)

    subgkeys = set(graphs.keys())
    for name, graph in graphs.items():
        core.switchGraph(name)
        loadGraph(graph['nodes'], subgkeys)

    
    applies = []
    nodes = graphs['main']['nodes']
    for ident, data in nodes.items():
        if 'special' in data:
            continue
        options = data['options']
        if 'OUT' in options or 'VIEW' in options:
            applies.append(ident)

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
                    core.setNodeParam(ident, name, value)
        ### ENDOF XINXIN HAPPY <<<<<

        core.frameBegin()
        while core.substepBegin():
            core.applyNodes(applies)
            core.substepEnd()
        core.frameEnd()

    print('EXITING')


def loadGraph(nodes, subgkeys):
    core.clearNodes()

    for ident, data in nodes.items():
        if 'special' in data:
            continue
        name = data['name']
        inputs = data['inputs']
        params = data['params']
        options = data['options']

        if name in subgkeys:
            params['name'] = name
            name = 'Subgraph'
        elif name == 'ExecutionOutput':
            name = 'Route'
        core.addNode(name, ident)

        for name, input in inputs.items():
            if input is None:
                continue
            srcIdent, srcSockName = input
            core.bindNodeInput(ident, name, srcIdent, srcSockName)

        for name, value in params.items():
            core.setNodeParam(ident, name, value)

        core.setNodeOptions(ident, set(options))

        core.completeNode(ident)


def dumpDescriptors():
    return core.dumpDescriptors()


__all__ = [
    'runScene',
    'dumpDescriptors',
]
