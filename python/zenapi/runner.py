import zen

from zenutils import run_script

zen.loadLibrary('libzenbase.so')
zen.loadLibrary('libzenvdb.so')
zen.loadLibrary('libOCTlib.so')
zen.loadLibrary('libFLIPlib.so')



def get_descriptors():
    descs = zen.dumpDescriptors()
    if isinstance(descs, bytes):
        descs = descs.decode()
    return descs


def run_graph(nodes, nframes, iopath):
    zen.setIOPath(iopath)
    for frameid in range(nframes):
        print('FRAME:', frameid)
        zen.frameBegin()
        while zen.substepShouldContinue():
            zen.substepBegin()
            run_graph_once(nodes)
            zen.substepEnd()
        zen.frameEnd()
    print('EXITING')


def run_graph_once(nodes):
    for ident in nodes:
        data = nodes[ident]
        name = data['name']
        inputs = data['inputs']
        params = data['params']

        print('APPLY', name, ident)
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
