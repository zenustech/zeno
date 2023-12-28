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
    std::string stdOutErr =
        "import sys\n\
class CatchOutErr:\n\
    def __init__(self):\n\
        self.value = ''\n\
    def write(self, txt):\n\
        self.value += txt\n\
    def flush(self):\n\
        pass\n\
catchOutErr = CatchOutErr()\n\
sys.stdout = catchOutErr\n\
sys.stderr = catchOutErr\n\
"; //this is python code to redirect stdouts/stderr

    Py_Initialize();
    PyObject* pModule = PyImport_AddModule("__main__"); //create main module
    PyRun_SimpleString(stdOutErr.c_str()); //invoke code to redirect

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = pModel->index("main");
    QModelIndex scriptIdx = pModel->paramIndex(subgIdx, index(), "script", true);
    ZASSERT_EXIT(scriptIdx.isValid());
    QString script = scriptIdx.data(ROLE_PARAM_VALUE).toString();

    //Py_Initialize();
    if (PyRun_SimpleString(script.toUtf8()) < 0) {
        zeno::log_warn("Python Script run failed");
    }
    PyObject* catcher = PyObject_GetAttrString(pModule, "catchOutErr"); //get our catchOutErr created above
    PyObject* output = PyObject_GetAttrString(catcher, "value"); //get the stdout and stderr from our catchOutErr object
    if (output != Py_None)
    {
        QString str = QString::fromStdString(_PyUnicode_AsString(output));
        QStringList lst = str.split('\n');
        for (const auto& line : lst)
        {
            if (!line.isEmpty())
            {
                if (zenoApp->isUIApplication())
                    ZWidgetErrStream::appendFormatMsg(line.toStdString());
            }
        }
    }
}
