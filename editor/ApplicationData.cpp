#include "ApplicationData.h"
#include <zs/ztd/format.h>

using namespace zs;


ApplicationData::ApplicationData(QObject *parent)
    : QObject(parent) {}

void ApplicationData::load_scene(QString str) const {
    ztd::print("load_scene: ", str.toStdString());
}

ApplicationData::~ApplicationData() = default;
