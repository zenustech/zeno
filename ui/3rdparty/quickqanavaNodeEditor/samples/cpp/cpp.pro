TEMPLATE    = app
TARGET      = test-cpp
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../src/quickqanava.pri)

SOURCES	+=  cpp_sample.cpp

HEADERS += cpp_sample.h

OTHER_FILES +=  cpp_sample.qml  \
                CustomNode.qml  \
                CustomEdge.qml  \
                CustomGroup.qml

RESOURCES   +=  cpp_sample.qrc

