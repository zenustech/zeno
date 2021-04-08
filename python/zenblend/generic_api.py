from .proc_launcher import execute_script

def node_graph_to_script(
        links: dict[tuple[str, str], tuple[str, str]],
        node_types: dict[str, str],
        wanted: set[str]):

    deps: dict[str, set[str]] = {}
    srcs: dict[str, dict[str, str]] = {}

    for node_name in node_types.keys():
        deps[node_name] = set()
        srcs[node_name] = {}

    for (dst_node, dst_sock), (src_node, src_sock) in links.items():
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
        applies.append(name)

    for name in wanted:
        touch(name)

    res = "def execute(frame):\n"
    res += "\timport zen\n"
    res += "\tif frame == 0: zen.addNode('EndFrame', 'endFrame')\n"
    for name in applies:
        res += "\tif frame == 0: zen.addNode('{}', '{}')\n".format(
                node_types[name], name)
        for socket_name, src_objname in srcs[name].items():
            res += "\tzen.setNodeInput('{}', '{}', '{}')\n".format(
                            name, socket_name, src_objname)
        res += "\tzen.applyNode('{}')\n".format(name)
    res += "\tzen.applyNode('endFrame')\n"

    return res
