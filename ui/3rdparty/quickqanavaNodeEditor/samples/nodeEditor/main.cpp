#include <QGuiApplication>
#include <QQmlApplicationEngine>

// QuickQanava headers
#include <QuickQanava>

using namespace qan;

int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    engine.addPluginPath(QStringLiteral("../QuickQanava/src")); // Necessary only for development when plugin is not installed to QTDIR/qml
    QuickQanava::initialize(&engine);
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
