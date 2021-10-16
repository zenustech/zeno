#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "ApplicationData.h"


int main(int argc, char **argv) {
    QApplication app(argc, argv);
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("applicationData", new ApplicationData);
    engine.load(QUrl("qrc:/main.qml"));

    return app.exec();
}
