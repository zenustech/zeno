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


__all__ = [
    'serializeScene',
]
