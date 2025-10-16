#pragma once

#ifndef __ZQML_PANEL_H__
#define __ZQML_PANEL_H__

#include <QtQuickWidgets/QQuickWidget>

class GraphModel;

class ZQmlPanel : public QQuickWidget
{
    Q_OBJECT
public:
    ZQmlPanel(QWidget* parent = nullptr);
    void reload();

protected:
    void focusInEvent(QFocusEvent* event);
    void focusOutEvent(QFocusEvent* event);
    void enterEvent(QEvent* event);
    void leaveEvent(QEvent* event);
};

class ZQmlGraphWidget : public QQuickWidget
{
    Q_OBJECT
public:
    ZQmlGraphWidget(GraphModel* pModel, QWidget* parent = nullptr);

protected:
    void focusInEvent(QFocusEvent* event);
};

#endif