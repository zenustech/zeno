#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>

class IGraphsModel;

class ZExpandableSection;

class ZenoPropPanel : public QWidget
{
	Q_OBJECT
public:
    ZenoPropPanel(QWidget* parent = nullptr);
    ~ZenoPropPanel();
    void reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    virtual QSize sizeHint() const override;
    virtual QSize minimumSizeHint() const override;

protected:
    void mousePressEvent(QMouseEvent* event);

public slots:
    void onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onParamEditFinish();
    void onInputEditFinish();

private:
    ZExpandableSection* paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);
    ZExpandableSection* inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);

    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_idx;
};

#endif