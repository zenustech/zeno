#ifndef __ZTIMELINE_H__
#define __ZTIMELINE_H__

#include <QtWidgets>

class ZSlider;
class ZLabel;

class ZTimeline : public QWidget
{
    Q_OBJECT
public:
    ZTimeline(QWidget* parent = nullptr);

signals:
    void playForward(bool bPlaying);
    void playForwardOneFrame();
    void playForwardLastFrame();
    int sliderValueChanged(int);

public slots:
    void onTimelineUpdate(int frameid);

private:
    ZSlider* m_slider;
};

#endif