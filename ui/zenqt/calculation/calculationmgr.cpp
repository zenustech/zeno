#include "calculationmgr.h"
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>
#include "viewport/displaywidget.h"
#include "zassert.h"
#include "util/uihelper.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zeno/core/common.h>
#include "model/graphsmanager.h"
#include "model/GraphsTreeModel.h"


CalcWorker::CalcWorker(QObject* parent) {
    auto& sess = zeno::getSession();
    sess.registerRunTrigger([=]() {
        run();
    });

    if (m_bReportNodeStatus) {
        sess.registerNodeCallback([=](zeno::ObjPath nodePath, bool bDirty, zeno::NodeRunStatus status) {
            NodeState state;
            state.bDirty = bDirty;
            state.runstatus = status;
            emit nodeStatusChanged(nodePath, state);
        });
    }
}

void CalcWorker::run() {
    auto& sess = zeno::getSession();

    zeno::GraphException::catched([&] {
        sess.run();
    }, *sess.globalError);
    sess.globalState->set_working(false);

    if (sess.globalError->failed()) {
        QString errMsg = QString::fromStdString(sess.globalError->getErrorMsg());
        NodeState state;
        state.bDirty = true;
        state.runstatus = zeno::Node_RunError;
        zeno::ObjPath path = sess.globalError->getNode();
        emit nodeStatusChanged(path, state);
        emit calcFinished(false, path, errMsg);
    }
    else {
        emit calcFinished(true, {}, "");
    }
}


CalculationMgr::CalculationMgr(QObject* parent)
    : QObject(parent)
    , m_bMultiThread(true)
    , m_worker(nullptr)
{
    m_worker = new CalcWorker(this);
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::started, m_worker, &CalcWorker::run);
    connect(m_worker, &CalcWorker::calcFinished, this, &CalculationMgr::onCalcFinished);
    connect(m_worker, &CalcWorker::nodeStatusChanged, this, &CalculationMgr::onNodeStatusReported);
}

void CalculationMgr::onNodeStatusReported(zeno::ObjPath uuidPath, NodeState state)
{
    GraphsTreeModel* pMainTree = zenoApp->graphsManager()->currentModel();
    if (pMainTree) {
        const QModelIndex targetNode = pMainTree->getIndexByUuidPath(uuidPath);
        if (targetNode.isValid()) {
            UiHelper::qIndexSetData(targetNode, QVariant::fromValue(state), ROLE_NODE_RUN_STATE);
        }
    }
}

void CalculationMgr::onCalcFinished(bool bSucceed, zeno::ObjPath nodeUuidPath, QString msg)
{
    //确保此时计算线程不再跑逻辑，这里暂时是代码上约束，也就是CalcWorker::run()走完就发信号。
    m_thread.quit();
    m_thread.wait();
    emit calcFinished(bSucceed, nodeUuidPath, msg);
}

void CalculationMgr::run()
{
    if (m_bMultiThread) {
        m_thread.start();
    }
}

void CalculationMgr::kill()
{
    zeno::getSession().globalState->set_working(false);
}

void CalculationMgr::registerRenderWid(DisplayWidget* pDisp)
{
    m_registerRenders.insert(pDisp);
    connect(this, &CalculationMgr::calcFinished, pDisp, &DisplayWidget::onCalcFinished);
    connect(pDisp, &DisplayWidget::render_objects_loaded, this, &CalculationMgr::on_render_objects_loaded);
}

void CalculationMgr::unRegisterRenderWid(DisplayWidget* pDisp) {
    m_loadedRender.remove(pDisp);
}

void CalculationMgr::on_render_objects_loaded()
{
    DisplayWidget* pWid = qobject_cast<DisplayWidget*>(sender());
    ZASSERT_EXIT(pWid);
    m_loadedRender.insert(pWid);
    if (m_loadedRender.size() == m_registerRenders.size())
    {
        //todo: notify calc to continue, if still have something to calculate.
    }
}
