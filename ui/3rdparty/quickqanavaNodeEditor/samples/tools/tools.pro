TEMPLATE    = app
TARGET      = test-tools
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../src/quickqanava.pri)

RESOURCES   += ./tools.qrc

SOURCES     +=  ./tools.cpp

HEADERS     +=  ./tools.qml

OTHER_FILES +=  tools.qml
