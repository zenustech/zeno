def node_graph_to_script(
        links: "list[tuple[str, str, str, str]]",
        nodes: "dict[str, tuple[str, dict[str, str], tuple[float, float]]]",
        wanted: "list[str]",
        **extra_kwargs):

    deps: dict[str, set[str]] = {}
    srcs: dict[str, dict[str, str]] = {}

    for node_name in nodes.keys():
        deps[node_name] = set()
        srcs[node_name] = {}

    for dst_node, dst_sock, src_node, src_sock in links:
        deps[dst_node].add(src_node)
        srcs[dst_node][dst_sock] = src_node + '::' + src_sock

    visited : set[str] = set()
    applies : list[str] = []

    def touch(name):
        if name in visited:
            return
        visited.add(name)
        for depname in deps[name]:
            touch(depname)
        if name != 'ExecutionOutput':
            applies.append(name)

    for name in wanted:
        touch(name)

    res = "def execute(frame):\n"
    res += "\timport zen\n"
    res += "\tif frame == 0: zen.addNode('EndFrame', 'endFrame')\n"
    for name in applies:
        node_type, node_params, node_uipos = nodes[name]
        res += "\tif frame == 0: zen.addNode('{}', '{}')\n".format(
                node_type, name)
        for socket_name, src_objname in srcs[name].items():
            res += "\tzen.setNodeInput('{}', '{}', '{}')\n".format(
                            name, socket_name, src_objname)
        for param_name, param_val in node_params.items():
            repr_param_val = repr(param_val)
            res += "\tzen.setNodeParam('{}', '{}', {})\n".format(
                            name, param_name, repr_param_val)
        res += "\tzen.applyNode('{}')\n".format(name)
    res += "\tzen.applyNode('endFrame')\n"

    return res
