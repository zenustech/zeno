#include "commandnode.h"
#include <zeno/extra/TempNode.h>
#include "launch/corelaunch.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>


CommandNode::CommandNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
}

CommandNode::~CommandNode()
{
}

Callback_OnButtonClicked CommandNode::registerButtonCallback(const QModelIndex& paramIdx)
{
    //todo: check whether there is commands input.
    if (paramIdx.data(ROLE_PARAM_NAME) == "source")
    {
        Callback_OnButtonClicked cb = [=]() {
            onGenerateClicked();
        };
        return cb;
    }
    return ZenoNode::registerButtonCallback(paramIdx);
}

ZGraphicsLayout* CommandNode::initCustomParamWidgets()
{
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZenoParamPushButton* pExecuteBtn = new ZenoParamPushButton("Execute", -1, QSizePolicy::Expanding);
    pExecuteBtn->setMinimumHeight(28);
    pHLayout->addSpacing(-1);
    pHLayout->addItem(pExecuteBtn);
    pHLayout->addSpacing(-1);
    connect(pExecuteBtn, SIGNAL(clicked()), this, SLOT(onExecuteClicked()));
    return pHLayout;
}

void CommandNode::onGenerateClicked()
{
    auto main = zenoApp->getMainWindow();
    ZASSERT_EXIT(main);

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();

    TIMELINE_INFO tinfo = main->timelineInfo();

    LAUNCH_PARAM launchParam;
    launchParam.beginFrame = tinfo.beginFrame;
    launchParam.endFrame = tinfo.endFrame;
    QString path = pModel->filePath();
    path = path.left(path.lastIndexOf("/"));
    launchParam.zsgPath = path;

    launchParam.projectFps = main->timelineInfo().timelinefps;
    launchParam.generator = this->nodeId();

    AppHelper::initLaunchCacheParam(launchParam);
    launchProgram(pModel, launchParam);
}

void CommandNode::onExecuteClicked()
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = pModel->index("main");
    QModelIndex commandsIdx = pModel->paramIndex(subgIdx, this->index(), "commands", true);
    ZASSERT_EXIT(commandsIdx.isValid());
    QString commands = commandsIdx.data(ROLE_PARAM_VALUE).toString();
    QStringList cmds = commands.split("\n", QString::SkipEmptyParts);
    for (QString cmd : cmds) {
        //api examples:
        // NewGraph("graphName")
        // RemoveGraph("graphName")
        // idA = AddNode("graphName", "LightNode", "<preset-id>")  preset-id is a ident generated by generator.
        // AddLink("graphName", "xxx-LightNode", "outparamname", "yyy-LightNode", "inparamname")
    }
}
