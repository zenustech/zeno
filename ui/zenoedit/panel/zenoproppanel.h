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
    void reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);

protected:
    void mousePressEvent(QMouseEvent* event);

public slots:
    void onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onLineEditFinish();

private:
    QGroupBox* paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);
    QGroupBox* inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);

    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_idx;
};

#endif