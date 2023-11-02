TEMPLATE    = app
TARGET      = test-edges
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../src/quickqanava.pri)

SOURCES     +=  edges.cpp

OTHER_FILES +=  edges.qml curved.qml endings.qml ortho.qml

RESOURCES   +=  edges.qrc

