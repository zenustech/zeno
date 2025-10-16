# Edges

## Creating Edges

Edges are managed from `Qan.Graph` (QML) or `qan::Graph` (C++) interface, usually in graph `Component.onCompleted()` handler:

- `#!js Qan.Edge Qan.Graph.insertEdge(Qan.Node src, Qan.Node dst)`: Insert a directed edge from source node to destination node.
- `#!js Qan.Edge Qan.Graph.removeEdge(Qan.Edge)`:

Connections between nodes could be added from QML using Qan.Graph.insertEdge() function, or from c++ with qan::Graph::insertEdge() method. Edges graphics appearance is configured trought their `style` property, more information on available edge style option is available in [Styles section ](styles.md)

``` cpp hl_lines="7"
Component.onCompleted: {	// Qan.Graph.Component.onCompleted()
    var n1 = graph.insertNode()
    n1.label = "Hello World"; n1.item.x=50; n1.item.y= 50
    var n2 = graph.insertNode()
    n2.label = "Node 2"; n2.item.x=200; n2.item.y= 125

    var e = graph.insertEdge(n1, n2);
    defaultEdgeStyle.lineType = Qan.EdgeStyle.Curved
}
```

![Default Edge](edges/edges-edge.png)

Edges appearance could be tuned by changing default styles properties directly from QML with global variables `defaultEdgeStyle`, see the [Style Management](styles.md) section for more options.

Edge could be moved by mouse when `qan::EdgeItem::draggable` property is set to true, edge source and destination nodes will be moved with the edge (this is not the default behavior).

## Visual creation of edges

### Visual Connectors

QuickQanava allow visual connection of node with the `Qan.VisualConnector` component. Default visual connector is configured in `Qan.Graph` component using the following properties:

- `Qan.Graph.connectorEnabled` (bool): Set to true to enable visual connection of nodes (default to false).

- `Qan.Graph.connectorEdgeColor` (color): Set the visual connector preview edge color (useful to have Light/Dark theme support, default to black).

- `Qan.Graph.connectorCreateDefaultEdge` (bool, default to true): When set to false, default visual connector does not use Qan.Graph.insertEdge() to create edge, but instead emit `connectorRequestEdgeCreation()` signal to allow user to create custom edge be calling a user defined method on graph (`connectorEdgeInserted()` is not emitted).

- `Qan.Graph.setConnectorSource` (node): Select the node that should host the current visual connector.

``` cpp hl_lines="5"
Qan.GraphView {
  id: graphView
  anchors.fill: parent
  graph : Qan.Graph {
    connectorEnabled: true
	connectorEdgeInserted: console.debug( "Edge inserted between " + src.label + " and  " + dst.label)
	connectorEdgeColor: "violet"
	connectorColor: "lightblue"
  } // Qan.Graph
} // Qan.GraphView
```

![Visual connector configuration](edges/edges-visual-connector-configuration.png)

The following notifications callbacks are available:

- Signal `Qan.Graph.connectorEdgeInserted(edge)`: Emitted when the visual connector has been dragged on a destination node or edge to allow specific user configuration on created edge.

- Signal `Qan.Graph.connectorRequestEdgeCreation(src, dst)`: Emitted when the visual connector is dragged on a target with `createDefaultEdge` set to false: allow creation of custom edge (or any other action) by the user. Example:

``` cpp hl_lines="8"
Qan.GraphView {
  id: graphView
  anchors.fill: parent
  graph : Qan.Graph {
    id: graph
    connectorEnabled: true
	connectorCreateDefaultEdge: false
	onConnectorRequestEdgeCreation: { 
	  // Do whatever you want here, for example: graph.insertEdge(src, dst)
	}
  } // Qan.Graph
} // Qan.GraphView
```

Preventing the visual connector to be shown for specific nodes (for example to force user to use out port to create topology) is possible by setting the node item `Qan.NodeItem.connectable' property to false (See `qan::NodeItem::connectable` header documentation).

Reference documentation: [qan::Connector interface](https://github.com/cneben/QuickQanava/blob/master/src/qanConnector.h)

### Custom Connectors

Default connector component `Qan.Graph.connector` could be replaced by a user defined `Qan.VisualConnector` to customize connector behavior in more depth. It is possible to add multiple visuals connectors on the same node, using a connector to generate specific topologies (create edges with different concrete types) or select targets visually. Such an advanced use of custom connectors is demonstrated in 'connector' sample: [https://github.com/cneben/QuickQanava/tree/master/samples/connector](https://github.com/cneben/QuickQanava/tree/master/samples/connector)



