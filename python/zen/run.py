import zen

from zenutils import run_script, load_library

load_library('libzenbase.so')
load_library('libzenvdb.so')
load_library('libOCTlib.so')
load_library('libFLIPlib.so')


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
                value = eval('f' + repr(value))
            zen.setNodeParam(ident, name, value)

        zen.initNode(ident)

    for ident in nodes:
        data = nodes[ident]
        if 'OUT' in data['options']:
            zen.applyNode(ident)


__all__ = ['runGraph', 'runGraphOnce']
