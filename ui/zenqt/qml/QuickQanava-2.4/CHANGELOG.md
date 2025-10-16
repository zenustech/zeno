# CHANGELOG

## 20231116 2.4.0:
- Rewrite the Qt5/6 CMake configuration.
- Remove all QUICKQANAVA_QT_MAJOR_VERSION and all compilation options, compilation is now always "static".
- Fix bugs in multiple selection dragging.
- Update https://github.com/oysteinmyrmo/bezier to `0.2.1`.
- Features:
  - Add dragging of nodes / groups by keyboard arrow keys.
  - Add custom QML group delegate support in `qan::Graph::insertGroup()`.

## 20230406 prev2.3.0:
- Fix a serious bug on multiple selection dragging and snap to grid.
- Fix a table node grouping issue when cols==1 or rows==1.

## 20221226 prev2.3.0:
- #185: Add horizontal or vertical constrain on node/group dragging, see new property `qan::NodeItem::dragOrientation`.

## 20221225 prev2.3.0:
- #185: Add support for "snap to grid" when moving node or groups.
- Add |qan::Graph`, `snapToGrid` and `snapToGridSize` properties.

## 20221225 v2.2.0:
- Release v2.2.0

## 20221215 v2.2.0:
- #183: Add a `multipleSelectionEnabled` property to `qan::Graph` to enable or disable multiple selection.

## 20221204 v2.2.0:
- #161: Add graph dynamic graph preview (move, center on, zoom on).

## 20221011 v2.2.0:
- #169: Create qan::RightResizer and qan::BottomResizer.
- By default, nodes and groups are resizable from their right and bottom borders.
- Refactor qan::BottomRightResizer completely.

## 20221002 v2.2.0:
- #167: Reduce memory use by 7%/8% when lot of edges are used.
- #167: Add partial edge selection support.
- #164: Change signature of qan::Graph::connectorRequestEdgeCreation().

## 20220824 v2*$.1.1:
- Change signature of two public API methods (public but presumably mainly used internally):
  - Rename qan::Graph::collectInerEdges() to qan::Graph::collectInnerEdges().
  - Rename qan::Graph::collectAncestorsDfs() to qan::Graph::collectAncestors().
- Add qan::Graph::collectNeighbours() method.
- Debug and rename qan::Flickable::fitInView() to fitContentInView().

## 20220817 v2.1.0:
- First Github release 
- Multiple bug fixes on internal topology backend and memory management.
- QuickQanava now build with Qt from Qt 5.15 to Qt 6.3 with g++, msvc and clang with both qmake and CMake.
- Add Qan.HeatMapPreview component that could be used to help user explore large graphs:

## 20220427 v2.0.0:
- Release v2.0.0.
- Simplify and clean GTpo framework (which is no longer usefull outside of QuickQanava).
- Simplify memory management, remove most c++ smartpointers (See #148)

## 20201205:
- Update copyright notice.
- Multiple bug fixes.

**20201205:** 
+ Update copyright notice.
+ Multiple bug fixes.

**20191110:** 
+ Complete rewrite of LineGrid, various cosmetic fixes and bugfixes (update to v0.15.0).
+ Breaking change: Qan.PointGrid support removed.

**20190825:** 
+ Maintenance release, add more flexible content creation methods in qan::Graph.

**20180930:** 
+ Add support for more edge end/start geometry, add support for orthogonal edge layout.
+ v0.11.0 pre1 CMake support is broken on Windows, please use qmake builds.
    
**20180930:** 
- Release v0.11.0 pre1.
- Fix a sever rendering bug on Windows/OSX (OGL issue).
- Add support for more edge end/start geometry, add support for orthogonal edge layout.
- CMake support is broken on Windows, use qmake builds.
    
**20180704:**

- Release 0.10.0 alpha.

**20180628:**

- Upgrade the topology backend to GTpo 3 (a totally refactored GTpo): no major API changes from a user POV.
- Remove all restricted hyper edge support.

**20180602:**

- DOC: Fix 'grouping nodes' section and add a 'custom group' paragraph.
- DOC: In 'graph' section, update Qan.GraphView documentation and a grid subsection.
- DOC: In 'Advanced use', add an introduction to C++ initialization.

**20180529:**

- Fix all tests in GTpo, increase coverage for groups.
- Reactivate qan::Graph::removeGroup() method (see groups sample).
- Minor fixes in topology and dataflow samples.

**20180527:**

- Add the cpp sample to demonstrate the C++ API, add a complete custom group implementation sample.
- BUGFIX: Node fixed resizing ratio modification was not taken into account for currently selected node.
- BUGFIX: Bottom right resizer is now hidden when current node is unselected.
- BUGFIX: Bottom right resizer item for nodes and group is now shown on top of selection item.
- BUGFIX: When the currently selected group selection policy is set to unselectable, unselect the group.
- Add minimumSize and resizable properties to Qan.GroupItem (see groups sample).
- Expose resizeHandlerOpacity, resizeHandlerRadius, resizeHandlerWidth and resizeHandlerSize properties from Qan.GraphView to allow
  complete customization of bottom right resizer handler (works for both nodes and groups).
- Add support for Group.labelEditorVisible and Group.expandButtonVisible in Qan.Group and Qan.RectGroupTemplate.
- Default color for selection items, bottom right resizer and visual node connector is now dodgerblue (previously darkblue).
- Add qan::NodeItem::setRect(const QRectF&) and qan::GroupItem(const QRectF&) shortcut methods.
- Various cosmetics fixes in qan::Graph::removeFromSelection() and qan::Graph::removeFromSelectionImpl<>().
- Major cleaning of the groups sample and demonstration of new group styling and resizing options.

**20180429:**

- Major documentation update (see Graph Data Model and Node Style.

**20180326:** 

- Add style support for gradient fill and glow effect. Reactivate style management (see style sample).
