Advanced Use (C++)
============================

Using from C++
------------------

QuickQanava rely on and underlying QML engine to efficiently draw visual content using QML scene graph (eventually with HW OpenGL acceleration). While the API is almost 100% usable from C++, a QML engine must be initialized in the background for rendering topology, a minimal QuickQanava initialization from QML should thus create graph view and a graph topology components:

``` cpp hl_lines="10"
// main.qml
import QuickQanava          2.0 as Qan
import "qrc:/QuickQanava"   as Qan

ApplicationWindow {
    title: "QuickQanava cpp API"
    Qan.GraphView {
        anchors.fill: parent
        graph : Qan.Graph {
            objectName: "graph"
            anchors.fill: parent
        } // Qan.Graph: graph
    }
}
```

On the C++ side, a QML engine should be initialized before starting the application event loop:

``` cpp hl_lines="7"
// main.cpp
int	main(int argc, char** argv)
{
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    QuickQanava::initialize(&engine);   // Mandatory
    engine.load(QUrl("qrc:/main.qml"));
    // Custom "synchronous" topology initialization should be added here    
    return app.exec();
}
```

It is then easy to access `#!js Qan.Graph{ objectName:"graph"}` QML component trough it's `#!cpp qan::Graph`  virtual interface:
``` cpp hl_lines="3"
  QPointer<CustomGraph> graph = nullptr;
  for (const auto rootObject : engine.rootObjects()) {
    graph = qobject_cast<CustomGraph*>(rootObject->findChild<QQuickItem *>("graph"));
    if (graph)
      break;
  }
  if (graph) {
    // Go on
  }
```

At this point, any modification should be done:

- Synchronously before application event loop is started using app.exec().
- In signals handlers reacting to `qan::Graph` or `qan::GraphView` events such as `nodeClicked()` or `groupClicked()` and so on (See [qan::Graph](https://github.com/cneben/QuickQanava/blob/master/src/qanGraph.h) and [qan::GraphView](https://github.com/cneben/QuickQanava/blob/master/src/qanGraphView.h) signals).
- Using behavior observers (See [Behaviors section](advanced.md#observation-of-topological-modifications)).

Please refer to the [cpp sample](https://github.com/cneben/QuickQanava/tree/master/samples/cpp) and more specifically [cpp_sample.cpp](https://github.com/cneben/QuickQanava/blob/master/samples/cpp/cpp_sample.cpp) for a sample about using `qan::Graph` topology related methods.

Defining Custom Topology
------------------

QuickQanava topology is described using GTpo library. Topology is modelled using non visual objects modelling node ([qan::Node](https://github.com/cneben/QuickQanava/blob/master/src/qanNode.h) connected by directed edges ([qan::Edge](https://github.com/cneben/QuickQanava/blob/master/src/qanEdge.h) eventually grouped in [qan::Group](https://github.com/cneben/QuickQanava/blob/master/src/qanGroup.h). These non-visual objects (called primitives) are mapped to QML QQuickItem based visual items. Concrete QQuickItem are generated on demand using QML QQmlComponent objects. 

QuickQanava provide default delegates for all primitives, but custom delegate could be specify by providing an argument for all primitive creation functions: `qan::Graph::insertNode()`, `qan::Graph::insertEdge()`, `qan::Graph::insertGroup()` and their QML counterpart in `Qan.Graph`. Simple custom delegate mapping is described in [custom nodes](nodes.md#defining-custom-nodes) and [custom groups](nodes.md#custom-groups) sections.

Primitives classes could be subclassed from c++ to provide specific customization. Mapping between non-visual topology primitive and their QtQuick counterpart is managed trought static singleton factories defined in `qan::Node`, `qan::Edge` and `qan::Group`. Theses factories are automatically called from `qan::Graph` when a primitive creation request happen: a visual item with a specific style is then automatically created.

Creation of a custom graph (MyGraph) with custom node (MyNode) and a dedicated visual item (MyNodeItem) could be achieved with the following architecture:

![Graph Class Diagram](advanced/class-custom-nodes.png)

Factories have to be redefined in primitive subclasses:

  - `#!cpp static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept`: Return a (usually singleton) QML component that will be used for primitive visual delegate.
  - `#!cpp static  qan::NodeStyle*     style() noexcept`: Return a (usually singleton) style used as primitive default style.

Custom content is then created from a specialized `qan::Graph` class:

``` cpp hl_lines="4"
// class MyGraph : public qan::Graph ...

qan::Node* MyGraph::insertMyNode() noexcept {
  return insertNode<MyNode>()
}
```

``` cpp 
// In a custom MyNode.cpp defining MyNode (inheriting from qan::Node)
QQmlComponent*  MyNode::delegate(QQmlEngine& engine) noexcept
{
static std::unique_ptr<QQmlComponent>   MyNode_delegate;
    if (!MyNode_delegate)
            CustomRectNode_delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/MyNode.qml");
    return CustomRectNode_delegate.get();
}

qan::NodeStyle* MyNode::style() noexcept
{
	static std::unique_ptr<qan::NodeStyle>  MyNode_style;
    if (!MyNode_style) {
        MyNode_style = std::make_unique<qan::NodeStyle>();
        MyNode_style->setBackColor(QColor("#ff29fc"));		// Initialize primitive default style here
    }
    return MyNode_style.get();
}
```

!!! note "Selection, visual connection and navigation will works out of the box for custom primitives (either nodes, edges or groups)."

Insertion of non Visual Content
------------------

Non visual edge or node could be used in graph to model complex topologies or add internal non-visual logic with:

- `#!cpp qan::Graph::insertNonVisualNode<>()`: default graph `nodeDelegate` will no be used, custom node `delegate()` may be oeither undefined or return nullptr.
- `#!cpp qan::Graph::insertNonVisualEdge<>(source, destination)`: Edge could be a regular node -> node edge or an oriented hyper edge node -> edge.


Observation of Topological Modifications
------------------

QuickQanava provide a full observation interface with the qan::Behaviour concept to react when underlying graph topology is modified. All primitives (nodes, edges or groups) could define custom behaviours to observe and react to topological changes.

- `qan::NodeBehaviour`: 

A behaviour could then be registered using: `registerBehaviour()` method in `qan::Node`.

``` cpp hl_lines="12 13"
#include <QuickQanava>

class CustomBehaviour : public qan::NodeBehaviour
{
  Q_OBJECT
public:
  explicit NodeBehaviour(QObject* parent = nullptr) :
     qan::NodeBehaviour{"Custom Behaviour", parent} { }
  virtual ~NodeBehaviour() override { /* Nil */ } 
  NodeBehaviour(const NodeBehaviour&) = delete;
protected:
  virtual void  inNodeInserted(qan::Node& inNode, qan::Edge& edge) noexcept override;
  virtual void  inNodeRemoved(qan::Node& inNode, qan::Edge& edge) noexcept override;
};
```

Such a custom node behaviour could be installed with the following code:

``` cpp hl_lines="4"
{
  qan::Graph graph;
  auto node = graph.insertNode();
  node->attachBehaviour(std::make_unique<CustomBehaviour>());
  // node will now react when an in node is inserted or removed
  auto source = graph.insertNode();
  auto edge = graph.insertEdge(source, node);   // CustomBehaviour::Inserted() called
  graph.removeEdge(edge);						// CustomBehaviour::inNodeRemoved() called
}
```

Methods `inNodeInserted()` and `inNodeRemoved()` are called automatically when an in node is inserted or removed on behaviour target node.

Reference documentation:

  - [qan::NodeBehaviour](https://github.com/cneben/QuickQanava/blob/master/src/qanBehaviour.h)
  - [qan::Node::installBehaviour()](https://github.com/cneben/QuickQanava/blob/425f1de0c75e1be85f51b90de517d75612978485/src/qanNode.h#L139)
  
