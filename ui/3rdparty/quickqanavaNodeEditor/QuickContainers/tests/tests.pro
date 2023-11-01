TEMPLATE    = app
TARGET      = qcmTests
CONFIG      += warn_on thread c++14
LANGUAGE    = C++
QT          +=  core gui qml quick charts testlib
INCLUDEPATH += ../src

include(../quickcontainers-common.pri)
include(../src/quickcontainers.pri)

win32-msvc*:INCLUDEPATH += $$GTEST_DIR/include $$GMOCK_DIR/include

HEADERS	+=  ./qcmTests.h                    \					\
            ./qcmContainerModelTests.h

SOURCES	+=  ./qcmTests.cpp                  \
            ./qcmContainerModelTests.cpp

CONFIG(debug, debug|release) {
    linux-g++*:     LIBS	+= -L../build/ -lgtest -lgmock
    win32-msvc*:    PRE_TARGETDEPS +=
    win32-msvc*:    LIBS	+= $$GTEST_DIR/msvc/x64/Debug/gtestd.lib $$GMOCK_DIR/msvc/x64/Debug/gmock.lib
    win32-g++*:     LIBS	+= -L../build/
}




