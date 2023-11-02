TEMPLATE    = app
TARGET      = test-groups
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2
INCLUDEPATH +=  ../../src
INCLUDEPATH +=  ../../QuickContainers/src

include(../../src/quickqanava.pri)

SOURCES     += ./groups.cpp
OTHER_FILES += ./groups.qml
RESOURCES   += ./groups.qrc
