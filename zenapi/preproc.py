import json
import zen


subgraph_loaded = set()


def preprocessGraph(nodes):
    for ident, data in nodes.items():
        name = data['name']
        if name == 'Subgraph':
            params = data['params']
            name = params['name']

            if name not in subgraph_loaded:
                with open(name, 'r') as f:
                    subg = json.load(f)

                zen.switchGraph(name)
                zen.loadGraph(subg)
                zen.switchGraph('main')

                subgraph_loaded.add(name)

    return nodes
