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
    QPair<int, int> fromTo() const;
    void initFromTo(int from, int to);
    void resetSlider();
    int value() const;

protected:
    void paintEvent(QPaintEvent* event) override;

signals:
    void playForward(bool bPlaying);
    void playForwardOneFrame();
    void playForwardLastFrame();
    void sliderValueChanged(int);

public slots:
    void onTimelineUpdate(int frameid);
    void onFrameEditted();
    void setSliderValue(int frameid);
    void setPlayButtonChecked(bool bToggle);
    void togglePlayButton(bool bOn);

private:
    void initStyleSheet();
    void initSignals();
    void initButtons();
    void initSize();

    int m_frames;

    Ui::Timeline *m_ui;
};

#endif