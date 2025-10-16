
CONFIG      += warn_on qt thread c++14
QT          += core widgets gui qml quick

include(quickcontainers/quickcontainers.pri)

# Note: QMake and pri inclusion is reserved to Qt5.

DEPENDPATH      += $$PWD
INCLUDEPATH     += $$PWD
RESOURCES       += $$PWD/QuickQanava_static.qrc

RESOURCES       += $$PWD/GraphicalEffects5/QuickQanavaGraphicalEffects.qrc

HEADERS +=  $$PWD/QuickQanava.h             \
            $$PWD/qanUtils.h                \
            $$PWD/qanGraphView.h            \
            $$PWD/qanEdge.h                 \
            $$PWD/qanEdgeItem.h             \
            $$PWD/qanNode.h                 \
            $$PWD/qanNodeItem.h             \
            $$PWD/qanPortItem.h             \
            $$PWD/qanSelectable.h           \
            $$PWD/qanDraggable.h            \
            $$PWD/qanAbstractDraggableCtrl.h\
            $$PWD/qanDraggableCtrl.h        \
            $$PWD/qanEdgeDraggableCtrl.h    \
            $$PWD/qanConnector.h            \
            $$PWD/qanBehaviour.h            \
            $$PWD/qanGroup.h                \
            $$PWD/qanTableGroup.h           \
            $$PWD/qanTableGroupItem.h       \
            $$PWD/qanTableCell.h            \
            $$PWD/qanTableBorder.h          \
            $$PWD/qanGroupItem.h            \
            $$PWD/qanGroupItem.cpp          \
            $$PWD/qanGraph.h                \
            $$PWD/qanGraph.hpp              \
            $$PWD/qanStyle.h                \
            $$PWD/qanStyleManager.h         \
            $$PWD/qanNavigable.h            \
            $$PWD/qanNavigablePreview.h     \
            $$PWD/qanAnalysisTimeHeatMap.h  \
            $$PWD/qanGrid.h                 \
            $$PWD/qanLineGrid.h             \
            $$PWD/qanBottomRightResizer.h   \
            $$PWD/qanRightResizer.h         \
            $$PWD/qanBottomResizer.h        \
            $$PWD/gtpo/container_adapter.h  \  # GTPO
            $$PWD/gtpo/edge.h               \
            $$PWD/gtpo/node.h               \
            $$PWD/gtpo/node.hpp             \
            $$PWD/gtpo/graph.h              \
            $$PWD/gtpo/graph.hpp            \
            $$PWD/gtpo/graph_property.h     \
            $$PWD/gtpo/observable.h         \
            $$PWD/gtpo/observer.h

SOURCES +=  $$PWD/qanGraphView.cpp          \
            $$PWD/qanUtils.cpp              \
            $$PWD/qanEdge.cpp               \
            $$PWD/qanEdgeItem.cpp           \
            $$PWD/qanNode.cpp               \
            $$PWD/qanNodeItem.cpp           \
            $$PWD/qanPortItem.cpp           \
            $$PWD/qanSelectable.cpp         \
            $$PWD/qanDraggable.cpp          \
            $$PWD/qanDraggableCtrl.cpp      \
            $$PWD/qanEdgeDraggableCtrl.cpp  \
            $$PWD/qanConnector.cpp          \
            $$PWD/qanBehaviour.cpp          \
            $$PWD/qanGraph.cpp              \
            $$PWD/qanGroup.cpp              \
            $$PWD/qanGroupItem.cpp          \
            $$PWD/qanTableGroup.cpp         \
            $$PWD/qanTableGroupItem.cpp     \
            $$PWD/qanTableCell.cpp          \
            $$PWD/qanTableBorder.cpp        \
            $$PWD/qanStyle.cpp              \
            $$PWD/qanStyleManager.cpp       \
            $$PWD/qanNavigable.cpp          \
            $$PWD/qanNavigablePreview.cpp   \
            $$PWD/qanAnalysisTimeHeatMap.cpp\
            $$PWD/qanGrid.cpp               \
            $$PWD/qanLineGrid.cpp           \
            $$PWD/qanBottomRightResizer.cpp \
            $$PWD/qanRightResizer.cpp       \
            $$PWD/qanBottomResizer.cpp

OTHER_FILES +=  $$PWD/../CHANGELOG.md               \
                $$PWD/QuickQanava                   \
                $$PWD/NavigablePreview.qml          \
                $$PWD/GraphPreview.qml              \
                $$PWD/HeatMapPreview.qml            \
                $$PWD/LineGrid.qml                  \
                $$PWD/GraphView.qml                 \
                $$PWD/Graph.qml                     \
                $$PWD/RectNodeTemplate.qml          \
                $$PWD/RectSolidBackground.qml       \
                $$PWD/RectSolidShadowBackground.qml \
                $$PWD/RectShadowEffect.qml          \
                $$PWD/RectSolidGlowBackground.qml   \
                $$PWD/RectGlowEffect.qml            \
                $$PWD/RectGradientBackground.qml        \
                $$PWD/RectGradientShadowBackground.qml  \
                $$PWD/RectGradientGlowBackground.qml    \
                $$PWD/CanvasNodeTemplate.qml        \
                $$PWD/Group.qml                     \
                $$PWD/RectGroupTemplate.qml         \
                $$PWD/TableGroup.qml                \
                $$PWD/TableCell.qml                 \
                $$PWD/BottomRightResizer.qml        \
                $$PWD/Node.qml                      \
                $$PWD/Port.qml                      \
                $$PWD/HorizontalDock.qml            \
                $$PWD/VerticalDock.qml              \
                $$PWD/Edge.qml                      \
                $$PWD/EdgeTemplate.qml              \
                $$PWD/EdgeStraightPath.qml          \
                $$PWD/EdgeOrthoPath.qml             \
                $$PWD/EdgeCurvedPath.qml            \
                $$PWD/EdgeSrcArrowPath.qml          \
                $$PWD/EdgeSrcCirclePath.qml         \
                $$PWD/EdgeSrcRectPath.qml           \
                $$PWD/EdgeDstArrowPath.qml          \
                $$PWD/EdgeDstCirclePath.qml         \
                $$PWD/EdgeDstRectPath.qml           \
                $$PWD/SelectionItem.qml             \
                $$PWD/VisualConnector.qml           \
                $$PWD/LabelEditor.qml               \
                $$PWD/qmldir_static
