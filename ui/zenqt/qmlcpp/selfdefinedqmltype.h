#ifndef SELFDEFINEDQMLTYPE_H
#define SELFDEFINEDQMLTYPE_H

#include <QSGSimpleRectNode>
#include <QtQuick/QQuickItem>

class SelfDefinedQMLType : public QQuickItem
{
    Q_OBJECT
public:
    SelfDefinedQMLType();
    ~SelfDefinedQMLType();

    Q_INVOKABLE void changeColor();

protected:
    QSGNode* updatePaintNode(QSGNode* node, UpdatePaintNodeData*);
    void keyPressEvent(QKeyEvent* event);

private:
    QSGSimpleRectNode* QSGSimNode;
};

#endif