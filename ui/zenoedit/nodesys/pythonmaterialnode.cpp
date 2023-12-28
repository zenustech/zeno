#include <Python.h>
#include "pythonmaterialnode.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>
#include "dialog/zmaterialinfosettingdlg.h"


PythonMaterialNode::PythonMaterialNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
}

PythonMaterialNode::~PythonMaterialNode()
{
}

ZGraphicsLayout* PythonMaterialNode::initCustomParamWidgets()
{
    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
    pHLayout->addSpacing(100);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Execute", -1, QSizePolicy::Expanding);
    pEditBtn->setMinimumHeight(32);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onExecuteClicked()));
    pVLayout->addLayout(pHLayout);

    pVLayout->addSpacing(10);

    ZGraphicsLayout* pHLayout1 = new ZGraphicsLayout(true);
    ZSimpleTextItem* pNameItem1 = new ZSimpleTextItem("MatchInfo");
    pNameItem1->setBrush(m_renderParams.socketClr.color());
    pNameItem1->setFont(m_renderParams.socketFont);
    pNameItem1->updateBoundingRect();

    pHLayout1->addItem(pNameItem1);

    pHLayout1->addSpacing(48);

    ZenoParamPushButton* pEditBtn1 = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pEditBtn1->setMinimumHeight(32);
    pHLayout1->addItem(pEditBtn1);
    connect(pEditBtn1, SIGNAL(clicked()), this, SLOT(onEditClicked()));
    pVLayout->addLayout(pHLayout1);
    return pVLayout;
}

void PythonMaterialNode::onExecuteClicked()
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
    QModelIndex subgIdx = subgIndex();
    //QModelIndex infoIdx = pModel->paramIndex(subgIdx, index(), "materialInfo", true);
    //ZASSERT_EXIT(infoIdx.isValid());
    //QString infoPath = infoIdx.data(ROLE_PARAM_VALUE).toString();
    //QModelIndex fileIdx = pModel->paramIndex(subgIdx, index(), "materialFile", true);
    //ZASSERT_EXIT(fileIdx.isValid());
    //QString filePath = fileIdx.data(ROLE_PARAM_VALUE).toString();
    //infoPath.replace('\\', '/');
    //filePath.replace('\\', '/');

    QString script = R"(
import json
import re
import zeno

json_data = {}
mat_data = {}
names_data = {}
keys_data = {}
match_data = {}
jsonStr = '%1'
json_data = json.loads(jsonStr)
if 'materialPath' in json_data:
    with open(json_data['materialPath'], 'r') as mat_file:
        mat_data = json.load(mat_file)
if 'materials' in json_data :
    names_data = json_data['materials']
else:
    print("materials is empty")
if 'keys' in json_data:
    keys_data = json_data['keys']
else:
    print("keys is empty")
if 'matchInfo' in json_data:
    match_data = json_data['matchInfo']
rows = int(len(names_data)**0.5)
cols = int(len(names_data) / rows if rows > 0 else 1)
pos = (0,0)
count = 0
for mat in names_data:
    subgName = ''
    for preSet, pattern in keys_data.items():
        if re.search(pattern, mat, re.I):
            subgName = preSet
            break
    if subgName == '':
        if "default" in keys_data:
            subgName = keys_data["default"]
    if subgName == '':
        print('Can not match ', mat)
    else:
        node = zeno.forkMaterial(preSet, mat, mat)
        if count > 0:
            row = int(count % rows)
            col = int(count / rows)
            newPos = (pos[0] + row * 600, pos[1]+col * 600)
            node.pos = newPos
        else:
            pos = node.pos
        count = count + 1
        if preSet in match_data and mat in mat_data:
            match = match_data[preSet]
            material = mat_data[mat]
            for k, v in match.items():
                if v in material:
                    setattr(node, k,material[v])
)";
    //Py_Initialize();
    m_jsonStr.replace("\r", "");
    m_jsonStr.replace("\n", "");
    if (PyRun_SimpleString(script.arg(m_jsonStr).toUtf8()) < 0) {
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


void PythonMaterialNode::onEditClicked()
{
    ZMaterialInfoSettingDlg::getMatchInfo(m_jsonStr);
}