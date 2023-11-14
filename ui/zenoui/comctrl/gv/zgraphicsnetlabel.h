#ifndef __ZGRAPHICS_NETLABEL_H__
#define __ZGRAPHICS_NETLABEL_H__

#include "zgraphicstextitem.h"
#include "../../nodesys/nodesys_common.h"

class ZGraphicsNetLabel : public ZGraphicsTextItem
{
    Q_OBJECT
public:
    ZGraphicsNetLabel(bool bInput, const QString& text, QGraphicsItem* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    QRectF boundingRect() const override;
    void SetTextInteraction(bool on, bool selectAll = false);
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
    enum {
        Type = ZTYPE_NETLABEL
    };
    int type() const override;
    QModelIndex paramIdx() const;

signals:
    void clicked();
    void aboutToDelete();
    void actionTriggered(QAction* pAction);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;
    void focusInEvent(QFocusEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;

private:
    const bool m_bInput;
};

#endif