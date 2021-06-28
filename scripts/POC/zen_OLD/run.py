import zen

from zenutils import run_script



def runGraph(nodes, nframes, iopath):
    zen.setIOPath(iopath)
    for frameid in range(nframes):
        print('FRAME:', frameid)
        zen.frameBegin()
        while zen.substepShouldContinue():
            zen.substepBegin()
            runGraphOnce(nodes)
            zen.substepEnd()
        zen.frameEnd()
    print('EXITING')


def evaluateExpr(expr):
    frame = zen.G.frameid
    return eval('f' + repr(expr))


def runGraphOnce(nodes):
    for ident in nodes:
        data = nodes[ident]
        name = data['name']
        inputs = data['inputs']
        params = data['params']

        zen.addNode(name, ident)

        for name, input in inputs.items():
            if input is None:
                continue
            srcIdent, srcSockName = input
            inputObjName = srcIdent + '::' + srcSockName
            zen.setNodeInput(ident, name, inputObjName)

        for name, value in params.items():
            if type(value) is str:
                value = evaluateExpr(value)
            zen.setNodeParam(ident, name, value)

        zen.initNode(ident)

    for ident in nodes:
        data = nodes[ident]
        if 'OUT' in data['options']:
            zen.applyNode(ident)


__all__ = ['runGraph', 'runGraphOnce']
