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
        const QModelIndex& nodeIdx,
        IGraphsModel* pModel,
        QObject* parent = nullptr);

    explicit PanelParamModel(
        const QModelIndex& nodeIdx,
        IGraphsModel* pModel,
        QObject* parent = nullptr);

    ~PanelParamModel();
    void initParams(NodeParamModel* nodeParams);
    void importPanelParam(const VPARAM_INFO& invisibleRoot);

public slots:
    void onNodeParamsInserted(const QModelIndex &parent, int first, int last);
    void onNodeParamsAboutToBeRemoved(const QModelIndex &parent, int first, int last);
};


#endif