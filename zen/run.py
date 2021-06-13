from . import core



def runGraph(nodes, nframes, iopath):
    #zen.setIOPath(iopath)
    for frameid in range(nframes):
        print('FRAME:', frameid)
        #zen.frameBegin()
        #while zen.substepShouldContinue():
            #zen.substepBegin()
        runGraphOnce(nodes, frameid)
            #zen.substepEnd()
        #zen.frameEnd()
    print('EXITING')


def evaluateExpr(expr, frame=None):
    return eval('f' + repr(expr))


def runGraphOnce(nodes, frame=None):
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

    for ident in nodes:
        data = nodes[ident]
        if 'OUT' in data['options']:
            core.applyNode(ident)

def dumpDescriptors():
    return core.dumpDescriptors()


__all__ = ['runGraph', 'runGraphOnce', 'dumpDescriptors']
