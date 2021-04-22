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

    lines.append('def substep():')
    lines.append('\tzen.substepBegin()')

    sortedIdents = topology_sort(nodes)
    for ident in sortedIdents:
        data = nodes[ident]
        name = data['name']
        inputs = data['inputs']
        params = data['params']

        lines.append('\tif zen.G.substepid == 0: zen.addNode({!r}, {!r})'
                .format(name, ident))

        for name, value in params.items():
            lines.append('\tzen.setNodeParam({!r}, {!r}, {!r})'
                    .format(ident, name, value))

        for name, input in inputs.items():
            if input is None:
                continue
            srcIdent, srcSockName = input
            input = srcIdent + '::' + srcSockName
            lines.append('\tzen.setNodeInput({!r}, {!r}, {!r})'
                    .format(ident, name, input))

        lines.append('\tzen.applyNode({!r})'
                .format(ident))

    lines.append('\tzen.substepEnd()')
    lines.append('')

    lines.append('def execute():')
    lines.append('\tzen.frameBegin()')
    lines.append('\twhile zen.substepShouldContinue():')
    lines.append('\t\tsubstep()')
    lines.append('\tzen.frameEnd()')

    return '\n'.join(lines)
