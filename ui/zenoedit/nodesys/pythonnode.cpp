#include <Python.h>
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
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZSimpleTextItem* pNameItem = new ZSimpleTextItem("    ");
    pNameItem->setBrush(m_renderParams.socketClr.color());
    pNameItem->setFont(m_renderParams.socketFont);
    pNameItem->updateBoundingRect();

    pHLayout->addItem(pNameItem);

    pHLayout->addSpacing(48);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Execute", -1, QSizePolicy::Expanding);
    pEditBtn->setMinimumHeight(32);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onExecuteClicked()));
    return pHLayout;
}

void PythonNode::onExecuteClicked()
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = pModel->index("main");
    QModelIndex scriptIdx = pModel->paramIndex(subgIdx, index(), "script", true);
    ZASSERT_EXIT(scriptIdx.isValid());
    QString script = scriptIdx.data(ROLE_PARAM_VALUE).toString();

    //Py_Initialize();
    if (PyRun_SimpleString(script.toUtf8()) < 0) {
        zeno::log_warn("Python Script run failed");
        return;
    }
}
