TEMPLATE    = app
TARGET      = test-selection
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2
INCLUDEPATH +=  ../../src
INCLUDEPATH +=  ../../QuickContainers/src

include(../../src/quickqanava.pri)

SOURCES     +=  ./selection.cpp
OTHER_FILES +=  ./selection.qml  \
                ./CustomSelectionItem.qml

RESOURCES   +=  ./selection.qrc
