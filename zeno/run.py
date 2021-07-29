import json

from .dll import core


def evaluateExpr(expr, frameid):
    frame = frameid
    return eval('f' + repr(expr))


def runScene(graphs, nframes, iopath):
    core.setIOPath(iopath)

    core.clearAllState()
    for cmd, *args in serializeScene(graphs):
        getattr(core, cmd)(*args)

    applies = []
    nodes = graphs['main']['nodes']
    for ident, data in nodes.items():
        if 'special' in data:
            continue
        options = data['options']
        if 'VIEW' in options:
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


def serializeScene(graphs):
    subgkeys = set(graphs.keys())
    for name, graph in graphs.items():
        yield 'switchGraph', name
        yield from serializeGraph(graph['nodes'], subgkeys)


def serializeGraph(nodes, subgkeys):
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
        yield 'addNode', name, ident

        for name, input in inputs.items():
            if input is None:
                continue
            srcIdent, srcSockName = input
            yield 'bindNodeInput', ident, name, srcIdent, srcSockName

        for name, value in params.items():
            yield 'setNodeParam', ident, name, value

        for name in options:
            yield 'setNodeOption', ident, name

        yield 'completeNode', ident


def dumpDescriptors():
    return core.dumpDescriptors()


__all__ = [
    'runScene',
    'dumpDescriptors',
]
