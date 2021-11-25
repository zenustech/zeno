#ifndef __LayerWidget_H__
#define __LayerWidget_H__

#include "framework.h"

class LayerTreeView : public QTreeView
{
    Q_OBJECT
    
    enum MOUSE_HINT
    {
        MOUSE_IN_TEXT,
        MOUSE_IN_VISIBLE,
        MOUSE_IN_LOCK,
        MOUSE_IN_OTHER,
    };

public:
    LayerTreeView(QWidget* parent = nullptr);
    QSize sizeHint() const;

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;

private:
    void updateHoverState(QPoint pos);

    MOUSE_HINT m_hoverObj;
};

class LayerWidget : public QWidget
{
    Q_OBJECT
public:
    LayerWidget(QWidget* parent = nullptr);

public slots:
    void resetModel();

private:
    LayerTreeView* m_pLayer;
    QStandardItemModel* m_model;
};

#endif