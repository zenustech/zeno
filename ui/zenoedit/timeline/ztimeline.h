#ifndef __ZTIMELINE_H__
#define __ZTIMELINE_H__

#include <QtWidgets>

class ZSlider;
class ZIconLabel;

namespace Ui
{
    class Timeline;
}

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
    void run(int, int);

public slots:
    void onTimelineUpdate(int frameid);
    void onFrameEditted();

private:
    void initStyleSheet();

    int m_frames;
    ZSlider* m_slider;
    QLineEdit* m_pFrameFrom;
    QLineEdit* m_pFrameTo;

    Ui::Timeline *m_ui;
};

#endif