#include "ZenoSceneObject.h"

ZenoSceneObject::ZenoSceneObject(QObject *parent)
    : QObject(parent)
{}

int ZenoSceneObject::reken_tijden_uit() {
    return 233;
}

ZenoSceneObject::~ZenoSceneObject() = default;
