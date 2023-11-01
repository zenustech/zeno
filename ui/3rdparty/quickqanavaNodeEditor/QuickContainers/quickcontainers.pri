
CONFIG      += warn_on qt thread c++14
QT          += core widgets gui qml

INCLUDEPATH += $$PWD/include

HEADERS +=  $$PWD/include/qcmContainerModel.h       \
            $$PWD/include/qcmAbstractContainer.h    \
            $$PWD/include/qcmAdapter.h              \
            $$PWD/include/qcmContainer.h            \
            $$PWD/include/QuickContainers.h

OTHER_FILES +=  $$PWD/QuickContainers

