#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>

class IGraphsModel;

class ZenoPropPanel : public QWidget
{
	Q_OBJECT
public:
    ZenoPropPanel(QWidget* parent = nullptr);
    ~ZenoPropPanel();
    void init(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);

protected:
    void mousePressEvent(QMouseEvent* event);
};

#endif