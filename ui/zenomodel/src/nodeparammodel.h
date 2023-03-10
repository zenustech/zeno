#ifndef __NODE_PARAMMODEL_H__
#define __NODE_PARAMMODEL_H__

#include "viewparammodel.h"

class DictKeyModel;

class NodeParamModel : public ViewParamModel
{
    Q_OBJECT
public:
    explicit NodeParamModel(const QPersistentModelIndex& subgIdx, const QModelIndex& nodeIdx, IGraphsModel* pModel, bool bTempModel = false, QObject* parent = nullptr);
    ~NodeParamModel();

    void clearParams();
    void clone(ViewParamModel* pModel) override;

    bool getInputSockets(INPUT_SOCKETS &inputs);
    bool getOutputSockets(OUTPUT_SOCKETS &outputs);
    bool getParams(PARAMS_INFO &params);
    void setInputSockets(const INPUT_SOCKETS &inputs);
    void setParams(const PARAMS_INFO &params);
    void setOutputSockets(const OUTPUT_SOCKETS &outputs);

    VParamItem* getInputs() const;
    VParamItem* getParams() const;
    VParamItem* getOutputs() const;
    QModelIndexList getInputIndice() const;
    QModelIndexList getParamIndice() const;
    QModelIndexList getOutputIndice() const;

    void setAddParam(
            PARAM_CLASS cls,
            const QString& name,
            const QString& type,
            const QVariant& deflValue,
            PARAM_CONTROL ctrl,
            QVariant ctrlProps = QVariant(),
            SOCKET_PROPERTY prop = SOCKPROP_NORMAL,
            DICTPANEL_INFO dictPanel = DICTPANEL_INFO(),
            const QString& toolTip = QString()
    );
    void removeParam(PARAM_CLASS cls, const QString& name);
    QVariant getValue(PARAM_CLASS cls, const QString& name) const;
    QModelIndex getParam(PARAM_CLASS cls, const QString& name) const;

    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;

    QModelIndex indexFromPath(const QString& path) override;

private:
    void initUI() override;
    QList<EdgeInfo> exportLinks(const PARAM_LINKS& links);
    EdgeInfo exportLink(const QModelIndex& linkIdx);
    QStringList sockNames(PARAM_CLASS cls) const;
    void onSubIOEdited(const QVariant& value, const VParamItem* pItem);
    void onLinkAdded(const VParamItem* pItem);
    void clearLinks(VParamItem* pItem);
    void onModelAboutToBeReset();
    void onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void initDictSocket(VParamItem* pItem, const SOCKET_INFO& inSocket);
    void exportDictkeys(DictKeyModel* pModel, DICTPANEL_INFO& panel);

    IGraphsModel* m_pGraphsModel;
    const QPersistentModelIndex m_subgIdx;

    VParamItem* m_inputs;
    VParamItem* m_params;
    VParamItem* m_outputs;

    bool m_bTempModel;      //temp model on edit param dialog, no actual operation to the graph.
};

#endif