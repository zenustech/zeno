#pragma once

#include <QObject>

class ZenoSceneObject : public QObject {
public:
    explicit ZenoSceneObject(QObject *parent = nullptr);
    virtual int reken_tijden_uit();
    virtual ~ZenoSceneObject();
};
