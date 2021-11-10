#ifndef __LayerWidget_H__
#define __LayerWidget_H__

#include "framework.h"

class LayerTreeView : public QTreeView
{
    Q_OBJECT
public:
    LayerTreeView(QWidget* parent = nullptr);
};

class LayerWidget : public QWidget
{
    Q_OBJECT
public:
    LayerWidget(QWidget* parent = nullptr);

private:
    LayerTreeView* m_pHeader;
    LayerTreeView* m_pBody;
};

#endif