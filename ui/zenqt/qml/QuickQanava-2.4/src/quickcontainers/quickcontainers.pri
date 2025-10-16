
CONFIG      += warn_on qt thread c++14
QT          += core widgets gui qml

INCLUDEPATH += $$PWD

HEADERS +=  $$PWD/qcmContainerModel.h       \
            $$PWD/qcmAbstractContainer.h    \
            $$PWD/qcmAdapter.h              \
            $$PWD/qcmContainer.h            \
            $$PWD/QuickContainers.h

OTHER_FILES +=  $$PWD/QuickContainers

