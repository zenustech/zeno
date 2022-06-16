#include "zenoplayer.h"
#include "../zenoedit/zenoapplication.h"
#include "../zenoedit/viewport/viewportwidget.h"
#include "../zenoedit/viewport/camerakeyframe.h"
#include "../zenoedit/launch/corelaunch.h"
#include "../zenoedit/launch/serialize.h"
#include "../zenoedit/model/graphsmodel.h"
#include <zenoui/util/jsonhelper.h>
#include <graphsmanagment.h>
#include <viewport/zenovis.h>

ZenoPlayer::ZenoPlayer(QWidget* parent)
    : QWidget(parent)
{
    resize(1000, 680);
    setMinimumSize(1000, 680);
    initUI();
    move((QApplication::desktop()->width() - width())/2,(QApplication::desktop()->height() - height())/2);
    QTimer::singleShot(10,this,[=]{showMaximized();});
    m_pTimerUpVIew = new QTimer;
    connect(m_pTimerUpVIew, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

ZenoPlayer::~ZenoPlayer()
{

}

void ZenoPlayer::initUI()
{
    m_pMenuBar = initMenu();

    m_pView = new ViewportWidget;

    m_pCamera_keyframe = new CameraKeyframeWidget;
    Zenovis::GetInstance().m_camera_keyframe = m_pCamera_keyframe;

    QVBoxLayout* layMain = new QVBoxLayout;
    layMain->setMargin(0);
    layMain->setSpacing(0);
    layMain->addWidget(m_pMenuBar);
    layMain->addWidget(m_pView,10);
    setLayout(layMain);
}

QMenuBar *ZenoPlayer::initMenu()
{
    QMenuBar *pMenuBar = new QMenuBar;
    QMenu *pFile = new QMenu(tr("File"));
    QAction *pAction = new QAction(tr("Open"), pFile);
    pAction->setCheckable(false);
    pAction->setShortcut(QKeySequence(tr("Ctrl+O")));
    connect(pAction, SIGNAL(triggered()), this, SLOT(slot_OpenFileDialog()));
    pFile->addAction(pAction);

    pMenuBar->addMenu(pFile);
    return pMenuBar;
}

void ZenoPlayer::slot_OpenFileDialog()
{
    QString filePath = QFileDialog::getOpenFileName(this,tr("Open"), "", tr("Zensim Graph File (*.zsg)\nAll Files (*)"));
    if (filePath.isEmpty())
        return;

    Zenovis::GetInstance().startPlay(false);
    m_pTimerUpVIew->stop();
    m_iFrameCount = 0;

    auto pGraphs = zenoApp->graphsManagment();
    pGraphs->clear();
    IGraphsModel *pModel = pGraphs->openZsgFile(filePath);
    if (!pModel){
        QMessageBox::warning(this,tr("Error"),QString(tr("Open %1 error!")).arg(filePath));
        return;
    }
    GraphsModel* pLegacy = qobject_cast<GraphsModel*>(pModel);

    launchProgram(pLegacy, 0, m_iMaxFrameCount);

    Zenovis::GetInstance().startPlay(true);
    m_pTimerUpVIew->start(m_iUpdateFeq);

}

void ZenoPlayer::updateFrame(const QString &action)
{
    if(m_iFrameCount >= m_iMaxFrameCount)
    {
        m_iFrameCount = 0;
        Zenovis::GetInstance().setCurrentFrameId(m_iFrameCount);
    }
    m_pView->update();
    m_iFrameCount++;
}
