def topology_sort(nodes):
    order = []
    visited = set()

    def touch(ident):
        if ident in visited:
            return
        visited.add(ident)

        data = nodes[ident]
        inputs = data['inputs']

        for name, input in inputs.items():
            if input is None:
                continue

            srcIdent, srcSockName = input
            touch(srcIdent)

        order.append(ident)

    for ident in nodes.keys():
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

        p('\tif zen.G.substepid == 0: zen.addNode({!r}, {!r})', name, ident)

        for name, value in params.items():
            p('\tzen.setNodeParam({!r}, {!r}, {!r})', ident, name, value)

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
    print(res)
    return res
