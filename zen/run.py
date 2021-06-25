import json

from . import core



def runScene(scene, nframes, iopath):
    core.setIOPath(iopath)
    for frameid in range(nframes):
        print('FRAME:', frameid)
        global frame; frame = frameid  # todo: xinxin happy
        core.frameBegin()
        while core.substepBegin():
            runSceneOnce(scene)
            core.substepEnd()
        core.frameEnd()
    print('EXITING')


def evaluateExpr(expr):
    return eval('f' + repr(expr))


def loadGraph(nodes, subgkeys):
    #core.clearNodes()

    for ident in nodes:
        data = nodes[ident]
        name = data['name']
        inputs = data['inputs']
        params = data['params']

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
            if type(value) is str:
                value = evaluateExpr(value)
            core.setNodeParam(ident, name, value)

        core.completeNode(ident)


def runSceneOnce(graphs):
    subgkeys = set(graphs.keys())
    for name, nodes in graphs.items():
        core.switchGraph(name)
        loadGraph(nodes, subgkeys)

    nodes = graphs['main']
    # 'main' graph use 'OUT' as applies, subgraphs use 'SubOutput' as applies
    applies = []
    for ident in nodes:
        data = nodes[ident]
        if 'OUT' in data['options']:
            applies.append(ident)

    core.switchGraph('main')
    core.applyNodes(applies)

def dumpDescriptors():
    return core.dumpDescriptors()


__all__ = [
    'runScene',
    'runSceneOnce',
    'dumpDescriptors',
]
