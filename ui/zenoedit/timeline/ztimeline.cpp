#include "ztimeline.h"
#include "zslider.h"
#include <comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>


//////////////////////////////////////////////
ZTimeline::ZTimeline(QWidget* parent)
    : QWidget(parent)
    , m_slider(nullptr)
    , m_pFrameFrom(nullptr)
    , m_pFrameTo(nullptr)
{
    setFocusPolicy(Qt::ClickFocus);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	QPalette pal = palette();
	pal.setColor(QPalette::Window, QColor(30, 30, 30));
	setAutoFillBackground(true);
	setPalette(pal);

    QHBoxLayout* pLayout = new QHBoxLayout;

    m_pFrameFrom = new QLineEdit;
    m_pFrameFrom->setText("0");
    m_pFrameFrom->setFixedSize(ZenoStyle::dpiScaledSize(QSize(50, 24)));
    connect(m_pFrameFrom, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));

    m_pFrameTo = new QLineEdit;
    m_pFrameTo->setFixedSize(ZenoStyle::dpiScaledSize(QSize(50, 24)));
    m_pFrameTo->setText("1");
    connect(m_pFrameTo, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));

    QPushButton* pRun = new QPushButton(tr("Run"));
    pRun->setProperty("cssClass", "grayButton");
    pRun->setFixedSize(ZenoStyle::dpiScaledSize(QSize(87, 24)));

    QPushButton* pKill = new QPushButton(tr("Kill"));
    pKill->setProperty("cssClass", "grayButton");
    pKill->setFixedSize(ZenoStyle::dpiScaledSize(QSize(87, 24)));

    ZIconLabel* plblForward = new ZIconLabel;
    plblForward->setIcons(QSize(20, 20), ":/icons/playforward.svg", ":/icons/playforward_hover.svg", ":/icons/playforward_on.svg", ":/icons/playforward_on_hover.svg", ":/icons/playforward.svg");

    ZIconLabel* plblBackward = new ZIconLabel;
    plblBackward->setIcons(QSize(20, 20), ":/icons/playbackward.svg", ":/icons/playbackward_hover.svg", ":/icons/playforward_on.svg", ":/icons/playforward_on_hover.svg", ":/icons/playbackward.svg");

    ZIconLabel* plblBackwardOneFrame = new ZIconLabel;
    plblBackwardOneFrame->setIcons(QSize(20, 20), ":/icons/playbackward_oneframe.svg", ":/icons/playbackward_oneframe_hover.svg");

    ZIconLabel* plblForwardOneFrame = new ZIconLabel;
    plblForwardOneFrame->setIcons(QSize(20, 20), ":/icons/playforward_oneframe.svg", ":/icons/playforward_oneframe_hover.svg");

    ZIconLabel* plblBackwardFirstFrame = new ZIconLabel;
    plblBackwardFirstFrame->setIcons(QSize(20, 20), ":/icons/playbackward_firstframe.svg", ":/icons/playbackward_firstframe_hover.svg");

    ZIconLabel* plblForwardLastFrame = new ZIconLabel;
    plblForwardLastFrame->setIcons(QSize(20, 20), ":/icons/playforward_lastframe.svg", ":/icons/playforward_lastframe_hover.svg");

    m_slider = new ZSlider;
    m_slider->setFromTo(0, 1);
    
    pLayout->addWidget(m_pFrameFrom);
    pLayout->addWidget(m_pFrameTo);
    pLayout->addWidget(pRun);
    pLayout->addWidget(pKill);
    pLayout->addWidget(plblBackwardFirstFrame);
    pLayout->addWidget(plblBackwardOneFrame);
    pLayout->addWidget(plblBackward);
    pLayout->addWidget(plblForward);
    pLayout->addWidget(plblForwardOneFrame);
    pLayout->addWidget(plblForwardLastFrame);
    pLayout->addWidget(m_slider);
    pLayout->setContentsMargins(20, 3, 20, 0);

    setLayout(pLayout);

    connect(plblForward, SIGNAL(toggled(bool)), this, SIGNAL(playForward(bool)));
    connect(m_slider, SIGNAL(sliderValueChange(int)), this, SIGNAL(sliderValueChanged(int)));
    connect(pRun, &QPushButton::clicked, this, [=]() {
        if (m_pFrameFrom->text().isEmpty() || m_pFrameTo->text().isEmpty())
            return;
        int frameFrom = m_pFrameFrom->text().toInt();
        int frameTo = m_pFrameTo->text().toInt();
        if (frameTo > frameFrom)
            emit run(frameFrom, frameTo);
    });
}

void ZTimeline::onTimelineUpdate(int frameid)
{
    bool blocked = m_slider->signalsBlocked();
    m_slider->blockSignals(true);
    m_slider->setSliderValue(frameid);
    m_slider->blockSignals(blocked);
}

void ZTimeline::onFrameEditted()
{
    if (m_pFrameFrom->text().isEmpty() || m_pFrameTo->text().isEmpty())
        return;
    int frameFrom = m_pFrameFrom->text().toInt();
    int frameTo = m_pFrameTo->text().toInt();
    if (frameTo > frameFrom)
        m_slider->setFromTo(frameFrom, frameTo);
}
