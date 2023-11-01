TEMPLATE    = app
TARGET      = containermodel
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../quickcontainers-common.pri)
include(../../src/quickcontainers.pri)

SOURCES	+=  qcmContainerModelSample.cpp
HEADERS	+=  qcmContainerModelSample.h

DISTFILES   +=  main.qml                    \
                containermodel.qml          \
                listreference.qml

RESOURCES   +=  containermodel.qrc



