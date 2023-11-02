TEMPLATE    = app
TARGET      = test-topology
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../src/quickqanava.pri)

RESOURCES   += ./topology.qrc           \
               ./theme/qan/qan-theme.qrc

SOURCES     +=  ./topology.cpp      \
                ./qanFaceNode.cpp

HEADERS     +=  ./qanFaceNode.h     \
                ./topology.qml

OTHER_FILES +=  topology.qml        \
                FaceNode.qml        \
                NodesListView.qml   \
                EdgesListView.qml   \
                ScreenshotPopup.qml
