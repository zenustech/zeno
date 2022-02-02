#include "graphsmanagment.h"
#include <io/zsgreader.h>
#include <model/graphsmodel.h>
#include <zenoio/reader/zsgreader.h>
#include <zenoio/acceptor/modelacceptor.h>
#include <zenoui/util/uihelper.h>
#include <io/zsgwriter.h>
#include <zeno.h>
#include <zeno/core/Session.h>
#include <extra/GlobalState.h>
#include "launch/serialize.h"
#include "launch/corelaunch.h"


GraphsManagment::GraphsManagment(QObject* parent)
    : QObject(parent)
    , m_model(nullptr)
{
#ifdef Q_OS_WIN
    LoadLibrary("zeno_ZenoFX.dll");
    LoadLibrary("zeno_oldzenbase.dll");
#else
    void* dp = nullptr;
    dp = dlopen("libzeno_ZenoFX.so", RTLD_NOW);
    if (dp == nullptr)
        return;
    dp = dlopen("libzeno_oldzenbase.so", RTLD_NOW);
    if (dp == nullptr)
        return;
#endif
}

GraphsModel* GraphsManagment::currentModel()
{
    return m_model;
}

GraphsModel* GraphsManagment::openZsgFile(const QString& fn)
{
    cleanIOPath();
    QTemporaryDir dir("zenvis-");
    dir.setAutoRemove(false);
    if (dir.isValid())
    {
        g_iopath = dir.path();
        QByteArray bytes = g_iopath.toLatin1();
        zeno::state = zeno::GlobalState();
        zeno::state.setIOPath(bytes.data());

        QFile file(fn);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return nullptr;

        bytes = file.readAll();
        QJsonDocument doc = QJsonDocument::fromJson(bytes);
        QJsonArray ret = serializeScene(doc["graph"].toObject());

        QJsonDocument doc2(ret);
        QString strJson(doc2.toJson(QJsonDocument::Compact));
        bytes = strJson.toUtf8();

        zeno::loadScene(bytes.data());
        zeno::switchGraph("main");

        m_model = new GraphsModel(this);
        const zeno::Session& sess = zeno::getSession();
        for (auto it = sess.defaultScene->graphs.begin(); it != sess.defaultScene->graphs.end(); it++)
        {
            SubGraphModel* pSubModel = new SubGraphModel(m_model);
            const std::string& name = it->first;
            pSubModel->setName(QString::fromStdString(name));
            for (auto it2 = it->second->nodes.begin(); it2 != it->second->nodes.end(); it2++)
            {
                it2->second;
                int j;
                j = 0;
            }
            m_model->appendSubGraph(pSubModel);
        }
        m_model->clear();
    }
    return m_model;
}

GraphsModel* GraphsManagment::importGraph(const QString& fn)
{
    GraphsModel *pModel = openZsgFile(fn);
    Q_ASSERT(pModel);
    m_model->setDescriptors(pModel->descriptors());
    for (int i = 0; i < pModel->rowCount(); i++)
    {
        SubGraphModel *pSubGraphModel = pModel->subGraph(i);
        QString name = pSubGraphModel->name();
        if (SubGraphModel *pExist = m_model->subGraph(name)) {
            //todo: reload
        } else {
            m_model->appendSubGraph(pSubGraphModel->clone(m_model));
        }
    }
    return m_model;
}

void GraphsManagment::reloadGraph(const QString& graphName)
{
    if (m_model)
        m_model->reloadSubGraph(graphName);
}

bool GraphsManagment::saveCurrent()
{
    if (!m_model || !m_model->isDirty())
        return false;

    int flag = QMessageBox::question(nullptr, "Save", "Save changes?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (flag & QMessageBox::Yes)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void GraphsManagment::clear()
{
    if (m_model) {
        m_model->clear();
        delete m_model;
        m_model = nullptr;
    }
}

void GraphsManagment::removeCurrent()
{
    if (m_model) {
        m_model->onRemoveCurrentItem();
    }
}

QList<QAction*> GraphsManagment::getCategoryActions(QPointF scenePos)
{
    NODE_CATES cates = m_model->getCates();
    QList<QAction*> acts;
    if (cates.isEmpty())
    {
        QAction* pAction = new QAction("ERROR: no descriptors loaded!");
        pAction->setEnabled(false);
        acts.push_back(pAction);
        return acts;
    }
    for (const NODE_CATE& cate : cates)
    {
        QAction* pAction = new QAction(cate.name);
        QMenu* pChildMenu = new QMenu;
        pChildMenu->setToolTipsVisible(true);
        for (const QString& name : cate.nodes)
        {
            QAction* pChildAction = pChildMenu->addAction(name);
            //todo: tooltip
            connect(pChildAction, &QAction::triggered, this, [=]() {
                onNewNodeCreated(name, scenePos);
                });
        }
        pAction->setMenu(pChildMenu);
        acts.push_back(pAction);
    }
    return acts;
}

void GraphsManagment::onNewNodeCreated(const QString& descName, const QPointF& pt)
{
    NODE_DESCS descs = m_model->descriptors();
    const NODE_DESC& desc = descs[descName];

    const QString& nodeid = UiHelper::generateUuid(descName);
    NODE_DATA node;
    node[ROLE_OBJID] = nodeid;
    node[ROLE_OBJNAME] = descName;
    node[ROLE_OBJTYPE] = NORMAL_NODE;
    node[ROLE_INPUTS] = QVariant::fromValue(desc.inputs);
    node[ROLE_OUTPUTS] = QVariant::fromValue(desc.outputs);
    node[ROLE_PARAMETERS] = QVariant::fromValue(desc.params);
    node[ROLE_OBJPOS] = pt;

    SubGraphModel* pModel = m_model->currentGraph();
    int row = pModel->rowCount();
    pModel->appendItem(node, true);
}

ZenoSubGraphScene* GraphsManagment::scene(const QString& subGraphName)
{
    return m_model->subGraph(subGraphName)->scene();
}