TEMPLATE    = app
TARGET      = test-style
CONFIG      += qt warn_on thread c++14
QT          += widgets core gui qml quick quickcontrols2 quickcontrols2

include(../../src/quickqanava.pri)

SOURCES     +=  ./style.cpp     \
                ./custom.cpp

HEADERS     +=  ./custom.h

OTHER_FILES += 	style.qml       \
                RectNode.qml    \
                RoundNode.qml	\
				CustomEdge.qml	

RESOURCES   += style.qrc
