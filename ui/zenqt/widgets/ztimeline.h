#ifndef __ZTIMELINE_H__
#define __ZTIMELINE_H__

#include <QtWidgets>
#include <zeno/core/common.h>

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
    void initFps(int fps);
    int fps();
    void resetSlider();
    int value() const;
    bool isPlayToggled() const;
    void updateKeyFrames(const QVector<int>& keys);
    void updateDopnetworkFrameCached(int frame);
    void updateDopnetworkFrameRemoved(int frame);
    void updateCachedFrame();
    void onSolverUpdate(zeno::SOLVER_MSG msg, int startFrame, int EndFrame);

protected:
    void paintEvent(QPaintEvent* event) override;

signals:
    void playForward(bool bPlaying);
    void playForwardOneFrame();
    void playForwardLastFrame();
    void sliderValueChanged(int);
    void sliderRangeChanged(int, int);

public slots:
    void onTimelineUpdate(int frameid);
    void onFrameEditted();
    void setSliderValue(int frameid);
    void setPlayButtonChecked(bool bToggle);
    void togglePlayButton(bool bOn);
    void stopSolver();

private:
    void initStyleSheet();
    void initSignals();
    void initButtons();
    void initSize();

    int m_frames;

    QScopedPointer<Ui::Timeline> m_ui;
};

#endif