#include "zenoapplication.h"
#include "zqmlpanel.h"
#include "model/GraphModel.h"
#include "model/graphsmanager.h"
#include <QQmlContext>


ZQmlPanel::ZQmlPanel(QWidget* parent)
    : QQuickWidget(zenoApp->getQmlEngine(), parent)
{
    //QQmlApplicationEngine* engine = zenoApp->getQmlEngine();
    GraphModel* pGraphM = zenoApp->graphsManager()->getGraph({ "main" });
    setResizeMode(QQuickWidget::SizeRootObjectToView);
    rootContext()->setContextProperty("nodesModel", pGraphM);

#if 1
    QQmlComponent component(this->engine(), QUrl(QStringLiteral("qrc:/ZGraphsView.qml")));
    if (component.status() != QQmlComponent::Ready) {
        qWarning() << "Error loading QML:";
        for (const QQmlError& error : component.errors()) {
            qWarning() << error.toString();
        }
        //return -1;
    }
#endif

    setSource(QUrl(QStringLiteral("qrc:/testQan.qml")));
}

void ZQmlPanel::reload()
{
    setSource(QUrl());
    engine()->clearComponentCache();
    setSource(QUrl::fromLocalFile("D:/zeno/ui/zenqt/qml/testQan.qml"));
}

void ZQmlPanel::focusInEvent(QFocusEvent* event)
{
    this->repaint();
    this->quickWindow();
    QQuickWidget::focusInEvent(event);
}

void ZQmlPanel::focusOutEvent(QFocusEvent* event)
{
    QQuickWidget::focusOutEvent(event);
}

void ZQmlPanel::enterEvent(QEvent* event)
{
    QQuickWidget::enterEvent(event);
}

void ZQmlPanel::leaveEvent(QEvent* event)
{
    QQuickWidget::leaveEvent(event);
}



ZQmlGraphWidget::ZQmlGraphWidget(GraphModel* pModel, QWidget* parent)
    : QQuickWidget(zenoApp->getQmlEngine(), parent)
{
    setResizeMode(QQuickWidget::SizeRootObjectToView);
    rootContext()->setContextProperty("nodesModel", pModel);
    setSource(QUrl(QStringLiteral("qrc:/testQan.qml")));
}

void ZQmlGraphWidget::focusInEvent(QFocusEvent* event)
{
    this->repaint();
    QQuickWidget::focusInEvent(event);
}