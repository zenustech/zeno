TEMPLATE    = app
TARGET      = resizer
CONFIG      += qt warn_on thread c++14
QT          += qml quick quickcontrols2

SOURCES +=  ../../src/qanBottomRightResizer.cpp \
            resizer.cpp

HEADERS +=  ../../src/qanBottomRightResizer.h

OTHER_FILES +=  resizer.qml
RESOURCES   +=  resizer.qrc


