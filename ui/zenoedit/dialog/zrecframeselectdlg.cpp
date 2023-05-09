#include "zrecframeselectdlg.h"
#include "ui_zrecframeselectdlg.h"
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Session.h>
#include "zassert.h"
#include "zenomainwindow.h"
#include "timeline/ztimeline.h"
#include "zenoapplication.h"

#define MAX_FRAME 100000



ZRecFrameSelectDlg::ZRecFrameSelectDlg(QWidget* parent)
    : QDialog(parent)
    , m_recStartF(0)
    , m_recEndF(0)
{
    m_ui = new Ui::RecFrameSelectDlg;
    m_ui->setupUi(this);

    const bool bWorking = zeno::getSession().globalState->working;
    int nRunFrames = zeno::getSession().globalComm->numOfFinishedFrame();
    auto pair = zeno::getSession().globalComm->frameRange();

    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    auto timelineframes = timeline->fromTo();

    connect(m_ui->editRecFrom, SIGNAL(editingFinished()), this, SLOT(onRecFromEdited()));
    connect(m_ui->editRecTo, SIGNAL(editingFinished()), this, SLOT(onRecToEdited()));
    connect(m_ui->btnRunFirst, SIGNAL(clicked()), this, SLOT(onRunNow()));
    connect(m_ui->btnRecLastRun, SIGNAL(clicked()), this, SLOT(onRecordLastRun()));
    connect(m_ui->btnRecordNow, SIGNAL(clicked()), this, SLOT(onRecordNow()));
    connect(m_ui->btnCancelRecord, SIGNAL(clicked()), this, SLOT(onCancelRecord()));

    if (!bWorking)
    {
        m_ui->btnRecordNow->setVisible(false);
        m_ui->btnRunFirst->setVisible(true);

        m_ui->editRecFrom->setValidator(new QIntValidator(0, MAX_FRAME));
        m_ui->editRecTo->setValidator(new QIntValidator(0, MAX_FRAME));

        if (nRunFrames == 0)
        {
            m_ui->lblRunFrame->setText(tr("The scene has not been run yet."));
            m_ui->btnRecLastRun->setVisible(false);

            m_ui->editRecFrom->setText(QString::number(timelineframes.first));
            m_ui->editRecTo->setText(QString::number(timelineframes.second));
        }
        else
        {
            int nLastRunFrom = pair.first;
            int nLastRunTo = nLastRunFrom + zeno::getSession().globalComm->maxPlayFrames() -1;

            m_ui->lblRunFrame->setText(tr("Last run frame: [%1 - %2]").arg(nLastRunFrom).arg(nLastRunTo));
            m_ui->btnRecLastRun->setVisible(true);

            m_ui->editRecFrom->setText(QString::number(nLastRunFrom));
            m_ui->editRecTo->setText(QString::number(nLastRunTo));
        }
    }
    else
    {
        int nRunStartF = pair.first;
        int nRunEndF = pair.second;

        m_ui->lblRunFrame->setText(tr("Running frame: [%1 - %2]").arg(nRunStartF).arg(nRunEndF));
        m_ui->btnRecLastRun->setVisible(false);
        m_ui->btnRunFirst->setVisible(false);
        m_ui->btnRecordNow->setVisible(true);

        m_ui->editRecFrom->setValidator(new QIntValidator(nRunStartF, nRunEndF));
        m_ui->editRecTo->setValidator(new QIntValidator(nRunStartF, nRunEndF));

        m_ui->editRecFrom->setText(QString::number(nRunStartF));
        m_ui->editRecTo->setText(QString::number(nRunEndF));
    }

    onRecFromEdited();
    onRecToEdited();
}

void ZRecFrameSelectDlg::onRecFromEdited()
{
    bool bOk = false;
    m_recStartF = m_ui->editRecFrom->text().toInt(&bOk);
    ZASSERT_EXIT(bOk);
}

void ZRecFrameSelectDlg::onRecToEdited()
{
    bool bOk = false;
    m_recEndF = m_ui->editRecTo->text().toInt(&bOk);
    ZASSERT_EXIT(bOk);
}

QPair<int, int> ZRecFrameSelectDlg::recordFrameRange(bool& runBeforeRun) const
{
    runBeforeRun = m_bRunBeforeRecord;
    return {m_recStartF, m_recEndF};
}

void ZRecFrameSelectDlg::onRunNow()
{
    if (!validateFrame())
        return;
    m_bRunBeforeRecord = true;
    accept();
}

void ZRecFrameSelectDlg::onRecordLastRun()
{
    if (!validateFrame())
        return;
    m_bRunBeforeRecord = false;
    accept();
}

void ZRecFrameSelectDlg::onRecordNow()
{
    if (!validateFrame())
        return;
    m_bRunBeforeRecord = false;
    accept();
}

void ZRecFrameSelectDlg::onCancelRecord()
{
    m_bRunBeforeRecord = false;
    reject();
}

bool ZRecFrameSelectDlg::validateFrame()
{
    if (m_recStartF > m_recEndF)
    {
        QMessageBox::information(this, tr("frame range"), tr("invalid frame range"));
        return false;
    }
    return true;
}