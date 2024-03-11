#include "calculationmgr.h"
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>


CalcWorker::CalcWorker(QObject* parent) {

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

