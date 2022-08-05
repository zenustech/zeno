#include "zenoplayer.h"
#include "../zenoedit/zenoapplication.h"
#include "../zenoedit/viewport/viewportwidget.h"
#include "../zenoedit/viewport/camerakeyframe.h"
#include "../zenoedit/launch/corelaunch.h"
#include "../zenoedit/launch/serialize.h"
#include "../zenoedit/model/graphsmodel.h"
#include <zenovis/DrawOptions.h>
#include <zenoui/util/jsonhelper.h>
#include <graphsmanagment.h>
#include <viewport/zenovis.h>
#include <cstdlib>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/utils/logger.h>


ZenoPlayer::ZenoPlayer(ZENO_PLAYER_INIT_PARAM param, QWidget *parent) 
    : QWidget(parent), m_InitParam(param) 
{
    setObjectName("ZenoPlayer");
    // resize(1000, 680);
    // setMinimumSize(1000, 680);
    initUI();

    if(!m_InitParam.sPixel.isEmpty())
    {
        QStringList tmpsPix = m_InitParam.sPixel.split("x");
        int pixw = tmpsPix.at(0).toInt();
        int pixh = tmpsPix.at(1).toInt();
        resize(pixw, pixh + m_pMenuBar->height() + 6);  // +6 UI interval
        m_pView->setCameraRes(QVector2D(pixw, pixh));
        m_pView->updatePerspective();
    }

    move((QApplication::desktop()->width() - width())/2,(QApplication::desktop()->height() - height())/2);
    // QTimer::singleShot(10,this,[=]{showMaximized();});
    m_pTimerUpVIew = new QTimer;

    if (m_InitParam.bRecord == true) {
        m_iMaxFrameCount = m_InitParam.iFrame;
    }

    connect(m_pTimerUpVIew, SIGNAL(timeout()), this, SLOT(updateFrame()));

    auto& inst = Zenovis::GetInstance();
    bool ret = connect(&inst, SIGNAL(frameDrawn(int)), this, SLOT(onFrameDrawn(int)));

    if (!m_InitParam.sZsgPath.isEmpty()) {
        startView(m_InitParam.sZsgPath);
    }
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
    layMain->addWidget(m_pView, 10);
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

    QMenu *pDisplay = new QMenu(tr("Display"));
    {
        QAction *pAction = new QAction(tr("Show Grid"), this);
        pAction->setCheckable(true);
        pAction->setChecked(true);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            Zenovis::GetInstance().getSession()->set_show_grid(pAction->isChecked());
            //todo: need a notify mechanism from zenovis/session.
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame();
        });

        pAction = new QAction(tr("Background Color"), this);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            auto [r, g, b] = Zenovis::GetInstance().getSession()->get_background_color();
            auto c = QColor::fromRgbF(r, g, b);
            c = QColorDialog::getColor(c);
            if (c.isValid()) {
                Zenovis::GetInstance().getSession()->set_background_color(c.redF(), c.greenF(), c.blueF());
                ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame();
            }
        });

        pDisplay->addSeparator();

        pAction = new QAction(tr("Smooth Shading"), this);
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            Zenovis::GetInstance().getSession()->set_smooth_shading(pAction->isChecked());
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame();
        });

        pAction = new QAction(tr("Normal Check"), this);
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            Zenovis::GetInstance().getSession()->set_normal_check(pAction->isChecked());
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame();
        });

        pAction = new QAction(tr("Wireframe"), this);
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            Zenovis::GetInstance().getSession()->set_render_wireframe(pAction->isChecked());
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame();
        });

        pDisplay->addSeparator();
        pAction = new QAction(tr("Solid"), this);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            const char *e = "bate";
            Zenovis::GetInstance().getSession()->set_render_engine(e);
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame(QString::fromLatin1(e));
        });
        pAction = new QAction(tr("Shading"), this);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            const char *e = "zhxx";
            Zenovis::GetInstance().getSession()->set_render_engine(e);
            Zenovis::GetInstance().getSession()->set_enable_gi(false);
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame(QString::fromLatin1(e));
        });
        pAction = new QAction(tr("VXGI"), this);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            const char *e = "zhxx";
            Zenovis::GetInstance().getSession()->set_render_engine(e);
            Zenovis::GetInstance().getSession()->set_enable_gi(true);
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame(QString::fromLatin1(e));
        });
        pAction = new QAction(tr("Optix"), this);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            const char *e = "optx";
            Zenovis::GetInstance().getSession()->set_render_engine(e);
            ((ZenoPlayer *)zenoApp->getWindow("ZenoPlayer"))->updateFrame(QString::fromLatin1(e));
        });
        pDisplay->addSeparator();

        pAction = new QAction(tr("Camera Keyframe"), this);
        pDisplay->addAction(pAction);

        pDisplay->addSeparator();

        pAction = new QAction(tr("English / Chinese"), this);
        pAction->setCheckable(true);
        pAction->setChecked(true);
        pDisplay->addAction(pAction);
    }

    QMenu *pRecord = new QMenu(tr("Record"));
    {
        QAction *pAction = new QAction(tr("Screenshot"), this);
        pAction->setShortcut(QKeySequence("F12"));
        pRecord->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            QString path = QFileDialog::getSaveFileName(
                nullptr, tr("Path to Save"), "",
                tr("PNG images(*.png);;JPEG images(*.jpg);;BMP images(*.bmp);;EXR images(*.exr);;HDR images(*.hdr);;"));
            QString ext = QFileInfo(path).suffix();
            int nsamples = 16;
            if (!path.isEmpty()) {
                Zenovis::GetInstance().getSession()->do_screenshot(path.toStdString(), ext.toStdString(), nsamples);
            }
        });
        pAction = new QAction(tr("Record Video"), this);
        pAction->setShortcut(QKeySequence(tr("Shift+F12")));
        pRecord->addAction(pAction);
    }

    QMenu *pEnvText = new QMenu(tr("EnvTex"));
    {
        QAction *pAction = new QAction(tr("BlackWhite"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Creek"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Daylight"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Default"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Footballfield"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Forest"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Lake"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Sea"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);
    }

    pMenuBar->addMenu(pFile);
    pMenuBar->addMenu(pDisplay);
    pMenuBar->addMenu(pRecord);
    pMenuBar->addMenu(pEnvText);
    return pMenuBar;
}

void ZenoPlayer::slot_OpenFileDialog()
{
    QString filePath = QFileDialog::getOpenFileName(this,tr("Open"), "", tr("Zensim Graph File (*.zsg)\nAll Files (*)"));
    if (filePath.isEmpty())
        return;

    startView(filePath);
}

void ZenoPlayer::updateFrame(const QString &action)
{
    auto &inst = Zenovis::GetInstance();
    auto sess = inst.getSession();
    if (sess) {
        auto scene = sess->get_scene();
        if (scene) {
            scene->drawOptions->num_samples = m_InitParam.bRecord ? m_InitParam.iSample : 16;
        }
    }

    if(m_iFrameCount >= m_iMaxFrameCount)
    {
        m_iFrameCount = 0;
        Zenovis::GetInstance().setCurrentFrameId(m_iFrameCount);
        if (m_InitParam.bRecord) {
            m_InitParam.bRecord = false;
            std::string cmd;
            cmd = QString("ffmpeg -r 24 -i %1 -c:v mpeg4 output.mp4")
                    .arg(m_InitParam.sPath+"/%07d.jpg").toStdString();
            std::puts(cmd.c_str());
            std::system(cmd.c_str());
            if (!m_InitParam.audioPath.isEmpty()) {
                cmd = QString("ffmpeg -i output.mp4 -i %1 -c:v copy -c:a aac output_av.mp4")
                        .arg(m_InitParam.audioPath).toStdString();
                std::puts(cmd.c_str());
                std::system(cmd.c_str());
            }

            QMessageBox::information(this, "Info","Video saved!");
        }
    }

    Zenovis::GetInstance().setCurrentFrameId(m_iFrameCount);
    m_pView->update();
    if(zeno::getSession().globalComm->maxPlayFrames()<=m_iFrameCount)
        return;
    //if (m_InitParam.bRecord == true) {
    //    QString path = QString("%1/%2.jpg").arg(m_InitParam.sPath).arg(m_iFrameCount, 7, 10, QLatin1Char('0'));
    //    QString ext = QFileInfo(path).suffix();
    //    int nsamples = m_InitParam.iSample;
    //    if (!path.isEmpty()) {
    //        //Zenovis::GetInstance().getSession()->do_screenshot(path.toStdString(), ext.toStdString(), nsamples);
    //    }
    //}

    m_iFrameCount++;
}

void ZenoPlayer::onFrameDrawn(int frameid)
{
    //zeno::log_info("onFrameDrawn is {}", frameid);

    if (m_InitParam.bRecord == true) {
        QString path = QString("%1/%2.jpg").arg(m_InitParam.sPath).arg(m_iFrameCount, 7, 10, QLatin1Char('0'));
        QString ext = QFileInfo(path).suffix();
        int nsamples = m_InitParam.iSample;
        if (!path.isEmpty()) {
            Zenovis::GetInstance().getSession()->do_screenshot(path.toStdString(), ext.toStdString(), nsamples);
        }
    }
}

void ZenoPlayer::startView(QString filePath) {
    Zenovis::GetInstance().startPlay(false);
    m_pTimerUpVIew->stop();
    m_iFrameCount = 0;

    auto pGraphs = zenoApp->graphsManagment();
    pGraphs->clear();
    IGraphsModel *pModel = pGraphs->openZsgFile(filePath);
    if (!pModel) {
        QMessageBox::warning(this, tr("Error"), QString(tr("Open %1 error!")).arg(filePath));
        return;
    }
//    auto &inst = Zenovis::GetInstance();
//    auto sess = inst.getSession();
//    if (sess) {
//        auto scene = sess->get_scene();
//        if (scene) {
//            scene->drawOptions->num_samples = m_InitParam.bRecord ? 1024 : 16;
//        }
//    }

    GraphsModel *pLegacy = qobject_cast<GraphsModel *>(pModel);

    launchProgram(pLegacy, 0, m_iMaxFrameCount);

    Zenovis::GetInstance().startPlay(true);
    m_pTimerUpVIew->start(m_iUpdateFeq);
}
