#include "subnetnode.h"
#include "model/graphsmodel.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <zenoui/util/uihelper.h>
#include "util/log.h"


class TypeValidator : public QValidator
{
public:
    explicit TypeValidator(QObject *parent = nullptr) : QValidator(parent) {

    }
    QValidator::State validate(QString& input, int&) const override
    {
        if (input.isEmpty())
        {
            return Acceptable;
        }
        if (input == "int" ||
            input == "string" ||
            input == "float" ||
            input == "bool" ||
            input == "vec3f" ||
            input == "curve" ||
            input == "heatmap")
        {
            return Acceptable;
        }
        else
        {
            return Intermediate;
        }
    }
    void fixup(QString& str) const override
    {
        str = "";
    }
};


SubnetNode::SubnetNode(bool bInput, const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
    , m_bInput(bInput)
{

}

SubnetNode::~SubnetNode()
{

}

void SubnetNode::onParamEditFinished(const QString& paramName, const QVariant& textValue)
{
    //get old name first.
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);
    const QString& nodeid = nodeId();
    QModelIndex subgIdx = this->subGraphIndex();
    const PARAMS_INFO& params = pModel->data2(subgIdx, index(), ROLE_PARAMETERS).value<PARAMS_INFO>();
    const QString& oldName = params["name"].value.toString();
    const QString& subnetName = pModel->name(subgIdx);
    if (oldName == textValue)
        return;

    ZenoNode::onParamEditFinished(paramName, textValue);
}

QValidator* SubnetNode::validateForParams(PARAM_INFO info)
{
    if (info.name == "type") {
        return new TypeValidator;
    }
    else {
        return ZenoNode::validateForParams(info);
    }
}
