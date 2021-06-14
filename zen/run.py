from . import core



def runGraph(nodes, nframes, iopath):
    #core.setIOPath(iopath)
    for frameid in range(nframes):
        print('FRAME:', frameid)
        #core.frameBegin()
        #while core.substepBegin():
        runGraphOnce(nodes, frameid)
            #core.substepEnd()
        #core.frameEnd()
    print('EXITING')


def evaluateExpr(expr, frame=None):
    return eval('f' + repr(expr))


def runGraphOnce(nodes, frame=None):
    core.clearNodes()

    for ident in nodes:
        data = nodes[ident]
        name = data['name']
        inputs = data['inputs']
        params = data['params']

        core.addNode(name, ident)

        for name, input in inputs.items():
            if input is None:
                continue
            srcIdent, srcSockName = input
            core.bindNodeInput(ident, name, srcIdent, srcSockName)

        for name, value in params.items():
            if type(value) is str:
                value = evaluateExpr(value, frame)
            core.setNodeParam(ident, name, value)

    applies = []
    for ident in nodes:
        data = nodes[ident]
        if 'OUT' in data['options']:
            applies.append(ident)

    core.applyNodes(applies)

def dumpDescriptors():
    return core.dumpDescriptors()


__all__ = ['runGraph', 'runGraphOnce', 'dumpDescriptors']
