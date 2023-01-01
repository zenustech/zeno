#ifndef __NODE_PARAMMODEL_H__
#define __NODE_PARAMMODEL_H__

#include "viewparammodel.h"

class NodeParamModel : public ViewParamModel
{
    Q_OBJECT
public:
    explicit NodeParamModel(const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent = nullptr);
    ~NodeParamModel();

    bool getInputSockets(INPUT_SOCKETS &inputs);
    bool getOutputSockets(OUTPUT_SOCKETS &outputs);
    bool getParams(PARAMS_INFO &params);
    void setInputSockets(const INPUT_SOCKETS &inputs);
    void setParams(const PARAMS_INFO &params);
    void setOutputSockets(const OUTPUT_SOCKETS &outputs);

    VParamItem* getInputs() const;
    VParamItem* getParams() const;
    VParamItem* getOutputs() const;

    void setParam(PARAM_CLASS cls,
            const QString& name,
            const QString& type,
            const QVariant& deflValue,
            SOCKET_PROPERTY prop);
    QVariant getValue(PARAM_CLASS cls, const QString& name) const;

    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

private:
    void initUI() override;
    QList<EdgeInfo> exportLinks(const PARAM_LINKS& links);
    EdgeInfo exportLink(const QModelIndex& linkIdx);

    VParamItem* m_inputs;
    VParamItem* m_params;
    VParamItem* m_outputs;
};

#endif