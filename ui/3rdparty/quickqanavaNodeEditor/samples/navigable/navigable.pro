TEMPLATE = app

QT += qml quick quickcontrols2

CONFIG += c++14

SOURCES +=  ./navigable.cpp                     \
            ../../src/qanNavigable.cpp          \
            ../../src/qanNavigablePreview.cpp   \
            ../../src/qanGrid.cpp               \
            ../../src/qanLineGrid.cpp

HEADERS += ../../src/qanNavigable.h         \
           ../../src/qanNavigablePreview.h  \
           ../../src/qanGrid.h              \
           ../../src/qanLineGrid.h

OTHER_FILES += ./navigable.qml                  \
               ./image.qml                      \
               ./grid.qml                       \
               ../../src/NavigablePreview.qml   \
               ../../src/LineGrid.qml

RESOURCES += navigable.qrc


