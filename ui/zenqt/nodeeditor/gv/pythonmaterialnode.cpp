#ifdef ZENO_WITH_PYTHON3
    #include <Python.h>
#endif
#include "pythonmaterialnode.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "model/graphsmanager.h"
#include "dialog/zmaterialinfosettingdlg.h"
#include <zeno/utils/log.h>


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
    ZSimpleTextItem* pNameItem1 = new ZSimpleTextItem("sync");
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
#ifdef ZENO_WITH_PYTHON3
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

    /*IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = subgIndex();*/
    MaterialMatchInfo info = getMatchInfo();
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
names_data = []
keys_data = {}
match_data = {}
names = '%1'
if names != '':
    names_data = names.split(',')
else:
    print("names is empty")
materialPath = '%2'
if materialPath != '':
    with open(materialPath, 'r') as mat_file:
        mat_data = json.load(mat_file)
keys = '%3'
if keys != '':
    keys_data = json.loads(keys)
else:
    print("key words is empty")
matchInfo = '%4'
if matchInfo != '':
    match_data = json.loads(matchInfo)
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
    if (PyRun_SimpleString(script.arg(info.m_names, info.m_materialPath, info.m_keyWords, info.m_matchInputs).toUtf8()) < 0) {
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
#else
    zeno::log_warn("The option 'ZENO_WITH_PYTHON3' should be ON");
#endif
}


void PythonMaterialNode::onEditClicked()
{
    MaterialMatchInfo info = getMatchInfo();
    ZMaterialInfoSettingDlg::getMatchInfo(info);
    setMatchInfo(info);
}

MaterialMatchInfo PythonMaterialNode::getMatchInfo()
{
    MaterialMatchInfo info;
    //TODO: 
#if 0
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = subgIndex();
    QModelIndex nameIdx = pModel->paramIndex(subgIdx, index(), "nameList", true);
    ZASSERT_EXIT(nameIdx.isValid(), info);
    info.m_names = nameIdx.data(ROLE_PARAM_VALUE).toString();

    QModelIndex keyIdx = pModel->paramIndex(subgIdx, index(), "keyWords", true);
    ZASSERT_EXIT(keyIdx.isValid(), info);
    info.m_keyWords = keyIdx.data(ROLE_PARAM_VALUE).toString();

    QModelIndex pathIdx = pModel->paramIndex(subgIdx, index(), "materialPath", true);
    ZASSERT_EXIT(pathIdx.isValid(), info);
    info.m_materialPath= pathIdx.data(ROLE_PARAM_VALUE).toString();

    QModelIndex matchIdx = pModel->paramIndex(subgIdx, index(), "matchInputs", true);
    ZASSERT_EXIT(matchIdx.isValid(), info);
    info.m_matchInputs = matchIdx.data(ROLE_PARAM_VALUE).toString();
#endif
    return info;
}

void PythonMaterialNode::setMatchInfo(const MaterialMatchInfo& matInfo)
{
#if 0
    PARAM_UPDATE_INFO info;
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = subgIndex();
    INPUT_SOCKETS inputs = index().data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    ZASSERT_EXIT(inputs.find("nameList") != inputs.end() && inputs.find("keyWords") != inputs.end()
        && inputs.find("matchInputs") != inputs.end() && inputs.find("materialPath") != inputs.end());
    const QString& nodeid = this->nodeId();
    //name list
    info.name = "nameList";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_names;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
    //key words
    info.name = "keyWords";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_keyWords;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
    //path
    info.name = "materialPath";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_materialPath;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
    //match inputs info
    info.name = "matchInputs";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_matchInputs;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
#endif
}