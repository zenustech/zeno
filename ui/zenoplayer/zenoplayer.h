#pragma once

#include <QtWidgets>
class ViewportWidget;
class CameraKeyframeWidget;

class ZenoPlayer : public QWidget
{
 Q_OBJECT
public:
    ZenoPlayer(QWidget* parent = nullptr);
    ~ZenoPlayer();



private:
    void initUI();
    QMenuBar* initMenu();

private slots:
    void slot_OpenFileDialog();
    void updateFrame(const QString& action = "");

private:
    QTimer* m_pTimerUpVIew;
    ViewportWidget *m_pView;
    CameraKeyframeWidget *m_pCamera_keyframe;
    QMenuBar* m_pMenuBar;

private:
    int m_iMaxFrameCount = 10;
    int m_iFrameCount = 0;
    int m_iUpdateFeq = 160;
};
