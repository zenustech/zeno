TEMPLATE    = app
TARGET      = test-connector
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2

include(../../src/quickqanava.pri)

SOURCES     +=  connector.cpp

OTHER_FILES +=  connector.qml   \
                default.qml     \
                custom.qml      \
                docks.qml       \
                customdocks.qml

RESOURCES   +=  connector.qrc

