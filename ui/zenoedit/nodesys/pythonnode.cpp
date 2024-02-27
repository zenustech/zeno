#include "pythonnode.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>


PythonNode::PythonNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{

}

PythonNode::~PythonNode()
{
}

ZGraphicsLayout* PythonNode::initCustomParamWidgets()
{
    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    pHLayout->addSpacing(100);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Execute", -1, QSizePolicy::Expanding);
    pEditBtn->setMinimumHeight(32);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onExecuteClicked()));

    ZGraphicsLayout* pHLayout1 = new ZGraphicsLayout(true);
    pHLayout1->addSpacing(100);
    ZenoParamPushButton* pGenerateBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
    pGenerateBtn->setMinimumHeight(32);
    pHLayout1->addItem(pGenerateBtn);
    connect(pGenerateBtn, SIGNAL(clicked()), this, SLOT(onGenerateClicked()));

    pVLayout->addLayout(pHLayout);
    pVLayout->addSpacing(10);
    pVLayout->addLayout(pHLayout1);
    return pVLayout;
}

void PythonNode::onExecuteClicked()
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = pModel->index("main");
    QModelIndex scriptIdx = pModel->paramIndex(subgIdx, index(), "script", true);
    ZASSERT_EXIT(scriptIdx.isValid());
    QString script = scriptIdx.data(ROLE_PARAM_VALUE).toString();
    QModelIndex argsIdx = pModel->paramIndex(subgIdx, index(), "args", true);
    QString args = argsIdx.data(ROLE_PARAM_VALUE).toString();
    if (!args.isEmpty())
    {
        rapidjson::Document doc;
        doc.Parse(args.toUtf8());

        if (!doc.IsObject()) {
            zeno::log_warn("document root not object: {}", std::string(args.toUtf8()));
        }
        else
        {
            auto objVal = doc.GetObject();
            for (auto iter = objVal.MemberBegin(); iter != objVal.MemberEnd(); iter++)
            {
                if (iter->value.IsString())
                    script = script.arg(iter->value.GetString());
                else if (iter->value.IsFloat())
                    script = script.arg(iter->value.GetFloat());
                else if (iter->value.IsInt())
                    script = script.arg(iter->value.GetInt());
                else
                    zeno::log_error("data type error");

            }
        }
    }
    AppHelper::pythonExcute(script);
}

void PythonNode::onGenerateClicked()
{
    AppHelper::generatePython(this->nodeId());
}
