#include <QApplication>
#include <QQmlApplicationEngine>
#include "ZenoSceneObject.h"

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    qmlRegisterType<ZenoSceneObject>("ZenusTech.Zeno", 1, 0, "ZenoSceneObject");

    QQmlApplicationEngine engine;
    engine.load(QUrl("qrc:/main.qml"));

    return app.exec();
}
