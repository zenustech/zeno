#include "calculationmgr.h"
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>
#include "viewport/displaywidget.h"
#include "zassert.h"


CalcWorker::CalcWorker(QObject* parent) {
    zeno::getSession().registerRunTrigger([=]() {
        run();
    });
}

void CalcWorker::run() {
    auto& sess = zeno::getSession();
    zeno::GraphException::catched([&] {
        sess.run();
    }, *sess.globalStatus);
    sess.globalState->set_working(false);
    if (sess.globalStatus->failed()) {
        std::string err = sess.globalStatus->toJson();
        emit calcFinished(false, QString::fromStdString(err));
    }
    else {
        emit calcFinished(true, "");
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
}

void CalculationMgr::onCalcFinished(bool bSucceed, QString msg)
{
    //确保此时计算线程不再跑逻辑，这里暂时是代码上约束，也就是CalcWorker::run()走完就发信号。
    m_thread.quit();
    m_thread.wait();
    emit calcFinished(bSucceed, msg);
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
