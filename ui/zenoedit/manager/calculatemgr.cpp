#include "calculatemgr.h"
#include "calcuate/calculateworker.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"


CalculateMgr::CalculateMgr(QObject* parent)
    : QObject(parent)
{
}

void CalculateMgr::doCalculate(const QString& json)
{
    if (m_thread.isRunning()) {
        //todo: kill or skip?
        return;
    }

    m_worker = new CalculateWorker;
    m_worker->moveToThread(&m_thread);
    connect(&m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(&m_thread, &QThread::started, m_worker, &CalculateWorker::work);
    connect(m_worker, &CalculateWorker::finished, &m_thread, &QThread::quit);

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    connect(m_worker, &CalculateWorker::viewUpdated, pWin, &ZenoMainWindow::updateViewport);
    m_worker->setProgJson(json);
    m_thread.start();
}