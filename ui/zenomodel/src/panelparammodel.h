#ifndef __PANEL_PARAMMODEL_H__
#define __PANEL_PARAMMODEL_H__

#include "viewparammodel.h"

class NodeParamModel;

class PanelParamModel : public ViewParamModel
{
    Q_OBJECT
public:
    explicit PanelParamModel(
        NodeParamModel* nodeParams,
        VPARAM_INFO root,
        IGraphsModel* pModel,
        QObject* parent = nullptr);

    explicit PanelParamModel(
        IGraphsModel* pModel,
        QObject* parent = nullptr);

    ~PanelParamModel();
    void initParams(NodeParamModel* nodeParams);
    void importPanelParam(const VPARAM_INFO& invisibleRoot);
    bool isDirty() const override;
    void markDirty() override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool isEditable(const QModelIndex& current) override;

public slots:
    void onNodeParamsInserted(const QModelIndex &parent, int first, int last);
    void onNodeParamsAboutToBeRemoved(const QModelIndex &parent, int first, int last);

private:
    bool m_bDirty;
};


#endif