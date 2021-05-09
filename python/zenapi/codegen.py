def topology_sort(nodes):
    order = []
    visited = set()
    portalDeps = {}

    for ident, data in nodes.items():
        if data['name'] == 'PortalIn':
            idname = data['params']['name']
            portalDeps[idname] = ident

    def touch(ident):
        if ident in visited:
            return
        visited.add(ident)

        data = nodes[ident]
        inputs = data['inputs']

        if data['name'] == 'PortalOut':
            idname = data['params']['name']
            srcIdent = portalDeps[idname]
            touch(srcIdent)

        for name, input in inputs.items():
            if input is None:
                continue

            srcIdent, srcSockName = input
            touch(srcIdent)

        order.append(ident)

    for ident, data in nodes.items():
        if 'options' in data:  # qt editor
            if 'OUT' in data['options']:
                touch(ident)
        else:  # blender editor
            if data['name'] == 'ExecutionOutput':
                touch(ident)

    return order


def generate_script(nodes):
    lines = []

    def p(fmt, *args):
        lines.append(fmt.format(*args))

    p('def substep():')
    p('\tzen.substepBegin()')

    sortedIdents = topology_sort(nodes)
    for ident in sortedIdents:
        data = nodes[ident]
        name = data['name']
        inputs = data['inputs']
        params = data['params']

        if name == 'ExecutionOutput':
            continue

        p('\tif zen.G.substepid == 0: zen.addNode({!r}, {!r})', name, ident)

        for name, value in params.items():
            if isinstance(value, str):
                valueRepr = 'f' + repr(value)
            else:
                valueRepr = repr(value)
            p('\tzen.setNodeParam({!r}, {!r}, {})', ident, name, valueRepr)

        for name, input in inputs.items():
            if input is None:
                continue
            srcIdent, srcSockName = input
            input = srcIdent + '::' + srcSockName
            p('\tzen.setNodeInput({!r}, {!r}, {!r})', ident, name, input)

        p('\tzen.applyNode({!r})', ident)

    p('\tzen.substepEnd()')
    p('')

    p('def execute():')
    p('\tzen.frameBegin()')
    p('\twhile zen.substepShouldContinue():')
    p('\t\tsubstep()')
    p('\tzen.frameEnd()')

    res = '\n'.join(lines)
    return res
