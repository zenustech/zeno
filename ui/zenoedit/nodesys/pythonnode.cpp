#include <Python.h>
#include "pythonnode.h"


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
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));
    return pHLayout;
}

void PythonNode::onEditClicked()
{
    Py_Initialize();
    if (PyRun_SimpleString("import ctypes") < 0) {
        zeno::log_warn("Failed to initialize Python module");
        return;
    }
}
