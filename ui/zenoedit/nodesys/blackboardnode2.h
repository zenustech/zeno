#ifndef __BLACKBOARD_NODE2_H__
#define __BLACKBOARD_NODE2_H__

#include "zenonode.h"

class BlackboardNode2 : public ZenoNode {
    Q_OBJECT
  public:
    BlackboardNode2(const NodeUtilParam &params, QGraphicsItem *parent = nullptr);
    ~BlackboardNode2();
    bool nodePosChanged(ZenoNode *);
    void onZoomed() override;
    QRectF boundingRect() const override;
    void onUpdateParamsNotDesc() override;
  protected:
    ZLayoutBackground *initBodyWidget(ZenoSubGraphScene *pScene) override;
    ZLayoutBackground *initHeaderWidget(IGraphsModel*) override;

    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

  private:
    bool isDragArea(QPointF pos);
    void updateBlackboard();
    void initUI();
    void updateView(bool isEditing);
    void updateClidItem(bool isAdd, const QString nodeId);
  private:
    bool m_bDragging;
    ZenoParamLineEdit *m_pTitle;
    ZenoParamBlackboard *m_pTextEdit;
    QGraphicsLinearLayout *m_mainLayout;
    ZenoSpacerItem *m_pMainSpaceItem;
};


#endif