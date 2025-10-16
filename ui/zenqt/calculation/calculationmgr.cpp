//#include <vld.h>
#include "calculationmgr.h"
#include <zeno/core/Session.h>
#include <zeno/extra/GraphException.h>
#include "viewport/displaywidget.h"
#include "zassert.h"
#include "util/uihelper.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zeno/core/common.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/extra/CalcContext.h>
#include <zeno/extra/GlobalState.h>
#include "model/graphsmanager.h"
#include "model/GraphsTreeModel.h"
#include "widgets/ztimeline.h"
#include "declmetatype.h"


CalcWorker::CalcWorker() {
    auto& sess = zeno::getSession();
    if (m_bReportNodeStatus) {
        sess.registerNodeCallback([=](std::string nodePath, bool bDirty, zeno::NodeRunStatus status) {
            QmlNodeRunStatus::Value q_status = static_cast<QmlNodeRunStatus::Value>(status);
            emit nodeStatusChanged(QString::fromStdString(nodePath), q_status);
            });
    }
}

void CalcWorker::setCurrentGraphPath(const QString& current_graph_path) {
    m_current_graph_path = current_graph_path;
}

void CalcWorker::run() {
    auto& sess = zeno::getSession();

    zeno::render_reload_info render_infos;
    sess.run("", render_infos); //如果局部运行子图，可能要考虑subinput的外部前置计算是否要计算
    sess.globalState->set_working(false);

    if (render_infos.error.failed()) {
        QString errMsg = QString::fromStdString(render_infos.error.getErrorMsg());
        QString path = QString::fromStdString(render_infos.error.getNode());
        emit nodeStatusChanged(path, QmlNodeRunStatus::RunError);
        emit calcFinished(false, path, errMsg, render_infos); //会发送到：DisplayWidget::onCalcFinished
    }
    else {
        emit calcFinished(true, {}, "", render_infos);
    }
}


CalculationMgr::CalculationMgr(QObject* parent)
    : QObject(parent)
    , m_bMultiThread(true)
    , m_worker(nullptr)
    , m_playTimer(new QTimer(this))
    , m_runstatus(RunStatus::NoRun)
{
    m_worker.reset(new CalcWorker);
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::started, m_worker.get(), &CalcWorker::run);
    connect(m_worker.get(), &CalcWorker::calcFinished, this, &CalculationMgr::onCalcFinished);
    connect(m_worker.get(), &CalcWorker::nodeStatusChanged, this, &CalculationMgr::onNodeStatusReported);
    connect(m_playTimer, SIGNAL(timeout()), this, SLOT(onPlayReady()));

    auto& sess = zeno::getSession();
    sess.registerRunTrigger([=]() {
        run();
    });
}

void CalculationMgr::onNodeStatusReported(QString qsPath, QmlNodeRunStatus::Value state)
{
    auto graphsMgr = zenoApp->graphsManager();
    GraphsTreeModel* pMainTree = graphsMgr->currentModel();
    if (pMainTree) {
        if (state == QmlNodeRunStatus::RunError) {
            //为了方便查看错误，整个子图路径的节点都要标红
            GraphModel* pMain = graphsMgr->getGraph({ "main" });
            QStringList pathList = qsPath.split('/', Qt::SkipEmptyParts);
            //把节点所在链路的所有子图（如果有）都mark
            while (pathList.size() > 1) {
                QString path = pathList.join('/');
                QModelIndex idx = graphsMgr->getNodeIndexByPath(path);
                UiHelper::qIndexSetData(idx, QVariant::fromValue(state), QtRole::ROLE_NODE_RUN_STATE);
                pathList.pop_back();
            }
        }
        else {
            const QModelIndex targetNode = pMainTree->getIndexByUuidPath(qsPath.toStdString());
            if (targetNode.isValid()) {
                UiHelper::qIndexSetData(targetNode, QVariant::fromValue(state), QtRole::ROLE_NODE_RUN_STATE);
                if (!m_bMultiThread) {
                    //TODO: 处理的时间里可能会包括改变节点状态和数据的操作，比如滑动时间轴，所以必须要控制事件的范围
                    //zenoApp->processEvents();
                }
            }
        }
    }
    if (state == QmlNodeRunStatus::RunSucceed) {
        auto pNode = zeno::getSession().getNodeByUuidPath(qsPath.toStdString());
        ZASSERT_EXIT(pNode);
        const QString& nodePath = QString::fromStdString(pNode->get_path());
        auto& pGlobalState = zeno::getSession().globalState;
        float total = pGlobalState->get_total_runtime();
        float consumed = pGlobalState->get_consume_time();
        if (total > 0) {
            if (consumed < total) {
                zenoApp->getMainWindow()->updateStatusTip(true, nodePath, consumed / total);
            }
            else if (consumed == total) {
                zenoApp->getMainWindow()->updateStatusTip(false, "Calcation Finish");
            }
        }
    }
}

void CalculationMgr::onCalcFinished(bool bSucceed, QString nodePath, QString msg, const zeno::render_reload_info& info)
{
    //确保此时计算线程不再跑逻辑，这里暂时是代码上约束，也就是CalcWorker::run()走完就发信号。
    if (m_bMultiThread)
    {
        m_thread.quit();
        m_thread.wait();
    }
    if (!bSucceed) {
        zeno::log_error("reportStatus: error in {}, message {}", nodePath.toStdString(), msg.toStdString());
    }
    setRunStatus(RunStatus::NoRun);
    if (bSucceed)
        zenoApp->getMainWindow()->updateStatusTip(false, "Calcation Finish");
    else
        zenoApp->getMainWindow()->updateStatusTip(false, "Calcation Fail");
    emit calcFinished(bSucceed, nodePath, msg, info);  //会发送到：DisplayWidget::onCalcFinished
}

void CalculationMgr::run()
{
    setRunStatus(RunStatus::Running);
    m_worker->setCurrentGraphPath(zenoApp->graphsManager()->currentGraphPath());
    if (m_bMultiThread) {
        m_thread.start();
    }
    else {
        m_worker->run();
    }
}

void CalculationMgr::onPlayReady() {
    auto& sess = zeno::getSession();
    if (!sess.is_auto_run()) {
        run();
    }
    //切到下一帧
    int frame = sess.globalState->getFrameId();
    sess.switchToFrame(frame + 1);

    //ui上也同步这一帧
    if (auto mainWin = zenoApp->getMainWindow()) {
        if (auto timeline = mainWin->timeline()) {
            timeline->blockSignals(true);
            timeline->setSliderValue(frame + 1);
            timeline->blockSignals(false);
        }
    }

    m_playTimer->start();
}

void CalculationMgr::onPlayTriggered(bool bToggled) {
    if (m_playTimer) {
        if (bToggled) {
            m_playTimer->start();
        }
        else {
            m_playTimer->stop();
        }

        if (auto mainWin = zenoApp->getMainWindow()) {
            if (auto timeline = mainWin->timeline()) {
                bool block = timeline->signalsBlocked();
                timeline->blockSignals(true);
                timeline->setPlayButtonChecked(bToggled);
                timeline->blockSignals(block);
            }
        }
    }
}

void CalculationMgr::onFrameSwitched(int frame) {
    //手动移动时间轴

    //停止播放
    m_playTimer->stop();
    if (auto mainWin = zenoApp->getMainWindow()) {
        if (auto timeline = mainWin->timeline()) {
            bool block = timeline->signalsBlocked();
            timeline->blockSignals(true);
            timeline->setPlayButtonChecked(false);
            timeline->blockSignals(block);
        }
    }

    auto& sess = zeno::getSession();
    sess.switchToFrame(frame);

    //如果有流体解算，直接取cache.
    //目前先不支持这种操作，原因是将流体解算和普通的图运算整合在一起，通过运行机制去获取，避免出现特例
    if (false)//!sess.is_auto_run() && !sess.get_solver().empty())
    {
        auto solvernode_path = sess.get_solver();
        zeno::NodeImpl* pSolverNode = sess.getNodeByUuidPath(solvernode_path);
        assert(pSolverNode);
        //zeno::CalcContext ctx;
        //如果frame没有解算结果，可能触发新的solver解算，要避免这一点。
        //pSolverNode->doApply(&ctx);
        //渲染更新：
        emit renderRequest(QString::fromStdString(solvernode_path));
        //emit calcFinished(true, solvernode_path, "");
    }
}

void CalculationMgr::kill()
{
    zeno::getSession().interrupt();
    zeno::getSession().globalState->set_working(false);
}

void CalculationMgr::clear() {
    zeno::getSession().markDirtyAndCleanResult();
    for (auto view : zenoApp->getMainWindow()->viewports())
        view->cleanUpScene();

    //还需要重设当前frame
    auto info = zenoApp->getMainWindow()->timelineInfo();
    zeno::getSession().globalState->updateFrameId(info.currFrame);
}

void CalculationMgr::run_and_clean() {
    // 清掉之前的分配记录
    //VLDGlobalDisable();
    //VLDGlobalEnable();  // 重新启用，开始新一轮记录
    auto& sess = zeno::getSession();

    setRunStatus(RunStatus::Running);
    m_worker->setCurrentGraphPath(zenoApp->graphsManager()->currentGraphPath());
    //不执行异步操作
    m_worker->run();
    clear();

#if 0 //TODO: 完善记录机制
    auto& objman = sess.objsMan;
    for (auto it = objman->m_rec_geoms.begin(); it != objman->m_rec_geoms.end(); it++) {
        auto pGeom = (zeno::GeometryObject_Adapter*)(*it);
        int j;
        j = 0;
    }
#endif

    Sleep(5000);
    //VLDReportLeaks();  // 会立即报告这个泄漏
}

void CalculationMgr::clearNodeObjs(const QModelIndex& nodeIdx)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(const_cast<QAbstractItemModel*>(nodeIdx.model()));
    if (pGraphM)
        pGraphM->clearNodeObjs(nodeIdx);
    bool isview = nodeIdx.data(QtRole::ROLE_NODE_ISVIEW).toBool();
    if (isview) {
        for (auto view : zenoApp->getMainWindow()->viewports()) {
            view->cleanUpScene();
        }
    }
}

void CalculationMgr::clearSubnetObjs(const QModelIndex& nodeIdx)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(const_cast<QAbstractItemModel*>(nodeIdx.model()));
    if (pGraphM)
        pGraphM->clearSubnetObjs(nodeIdx);
}

RunStatus::Value CalculationMgr::getRunStatus() const
{
    return m_runstatus;
}

void CalculationMgr::setRunStatus(RunStatus::Value runstatus)
{
    if (m_runstatus == runstatus) {
        return;
    }
    m_runstatus = runstatus;
    emit runStatus_changed();
}

bool CalculationMgr::getAutoRun() const
{
    auto& sess = zeno::getSession();
    return sess.is_auto_run();
}

void CalculationMgr::setAutoRun(bool autoRun)
{
    if (getAutoRun() == autoRun)
        return;
    auto& sess = zeno::getSession();
    sess.set_auto_run(autoRun);
    emit autorun_changed();
}

void CalculationMgr::registerRenderWid(DisplayWidget* pDisp)
{
    m_registerRenders.insert(pDisp);
    connect(this, &CalculationMgr::calcFinished, pDisp, &DisplayWidget::onCalcFinished);
    connect(this, &CalculationMgr::renderRequest, pDisp, &DisplayWidget::onRenderRequest);
    connect(pDisp, &DisplayWidget::render_reload_finished, this, &CalculationMgr::on_render_objects_loaded);
}

void CalculationMgr::unRegisterRenderWid(DisplayWidget* pDisp) {
    m_loadedRender.remove(pDisp);
}

bool CalculationMgr::isMultiThreadRunning() const {
    return m_bMultiThread;
}

void CalculationMgr::on_render_objects_loaded()
{
    DisplayWidget* pWid = qobject_cast<DisplayWidget*>(sender());
    ZASSERT_EXIT(pWid);
    m_loadedRender.insert(pWid);
    if (m_loadedRender.size() == m_registerRenders.size())
    {
        int j;
        j = 0;
        //todo: notify calc to continue, if still have something to calculate.
    }
}
