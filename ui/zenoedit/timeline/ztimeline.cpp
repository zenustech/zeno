#include "ztimeline.h"
#include "zslider.h"
#include <comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>


//////////////////////////////////////////////
ZTimeline::ZTimeline(QWidget* parent)
    : QWidget(parent)
    , m_slider(nullptr)
{
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	QPalette pal = palette();
	pal.setColor(QPalette::Window, QColor(30, 30, 30));
	setAutoFillBackground(true);
	setPalette(pal);

    QHBoxLayout* pLayout = new QHBoxLayout;

    QLineEdit* pFrame = new QLineEdit;
    pFrame->setText("1");
    pFrame->setFixedSize(ZenoStyle::dpiScaledSize(QSize(50, 24)));

    QPushButton* pRun = new QPushButton(tr("Run"));
    pRun->setObjectName("grayButton");
    pRun->setFixedSize(ZenoStyle::dpiScaledSize(QSize(87, 24)));

    QPushButton* pKill = new QPushButton(tr("Kill"));
    pKill->setObjectName("grayButton");
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
    
    pLayout->addWidget(pFrame);
    pLayout->addWidget(pRun);
    pLayout->addWidget(pKill);
    pLayout->addWidget(plblBackwardFirstFrame);
    pLayout->addWidget(plblBackwardOneFrame);
    pLayout->addWidget(plblBackward);
    pLayout->addWidget(plblForward);
    pLayout->addWidget(plblForwardOneFrame);
    pLayout->addWidget(plblForwardLastFrame);
    pLayout->addWidget(m_slider);

    setLayout(pLayout);

    connect(plblForward, SIGNAL(toggled(bool)), this, SIGNAL(playForward(bool)));
    connect(m_slider, SIGNAL(sliderValueChange(int)), this, SIGNAL(sliderValueChanged(int)));
    connect(pRun, &QPushButton::clicked, this, [=]() {
        int frames = pFrame->text().toInt();
        emit run(frames);
    });
}

void ZTimeline::onTimelineUpdate(int frameid)
{
    m_slider->setSliderValue(frameid);
}
