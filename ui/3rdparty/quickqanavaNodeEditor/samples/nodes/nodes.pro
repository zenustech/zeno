TEMPLATE    = app
TARGET      = test-nodes
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../src/quickqanava.pri)

SOURCES	+=  nodes.cpp

OTHER_FILES +=  nodes.qml       \
                default.qml     \
                custom.qml      \
                CustomNode.qml  \
                ControlNode.qml \
                DiamondNode.qml

RESOURCES   +=  nodes.qrc

