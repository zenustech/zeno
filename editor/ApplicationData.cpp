#include "ApplicationData.h"


ApplicationData::ApplicationData(QObject *parent)
    : QObject(parent) {}

Q_INVOKABLE void ApplicationData::load_scene(QString str) const {
    printf("load_scene %s\n", str.toStdString().c_str());
}

ApplicationData::~ApplicationData() = default;
