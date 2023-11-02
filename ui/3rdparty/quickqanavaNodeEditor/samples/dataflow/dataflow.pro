TEMPLATE    = app
TARGET      = test-dataflow
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../src/quickqanava.pri)

SOURCES     +=  dataflow.cpp    \
                qanDataFlow.cpp
				
HEADERS     +=  qanDataFlow.h

OTHER_FILES +=  dataflow.qml        \
                FlowNode.qml        \
                PercentageNode.qml  \
                OperationNode.qml   \
                ImageNode.qml       \
                TintNode.qml        \
                ColorNode.qml       \
                ColorPopup.qml

RESOURCES   +=  dataflow.qrc

