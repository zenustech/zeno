#pragma once

#include <QtWidgets>
class ViewportWidget;
class CameraKeyframeWidget;

struct ZENO_PLAYER_INIT_PARAM {
    QString sZsgPath = "";
    bool bRecord = false;
    int iFrame = 0;
    int iSFrame = 0;
    int iSample = 0;
    int iBitrate = 0;
    int iFps = 0;
    QString sPixel = "";
    QString sPath = "";
    QString audioPath = "";
    QString configFilePath = "";
    bool exitWhenRecordFinish = false;
};

class ZenoPlayer : public QWidget
{
 Q_OBJECT
public:
    ZenoPlayer(ZENO_PLAYER_INIT_PARAM param, QWidget *parent = nullptr);
    ~ZenoPlayer();

private:
    void initUI();
    QMenuBar* initMenu();

private slots:
    void slot_OpenFileDialog();
    void updateFrame(const QString& action = "");
    void onFrameDrawn(int frameid);
    void startView(QString filePath);

private:
    QTimer* m_pTimerUpVIew;
    ViewportWidget *m_pView;
    CameraKeyframeWidget *m_pCamera_keyframe;
    QMenuBar* m_pMenuBar;

private:
    int m_iMaxFrameCount = 10;
    int m_iFrameCount = 0;
    int m_iUpdateFeq = 160;

    ZENO_PLAYER_INIT_PARAM m_InitParam;
};
