#include "ztimeline.h"
#include "zslider.h"
#include <comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/effect/innershadoweffect.h>
#include <zeno/utils/envconfig.h>
#include <zenomodel/include/uihelper.h>
#include "../viewport/zenovis.h"
#include "ui_ztimeline.h"


//////////////////////////////////////////////
ZTimeline::ZTimeline(QWidget* parent)
    : QWidget(parent)
{
    m_ui = new Ui::Timeline;
    m_ui->setupUi(this);
 
    setFocusPolicy(Qt::ClickFocus);
    //QPalette pal = palette();
    //pal.setColor(QPalette::Window, QColor(42, 42, 42));
    //setAutoFillBackground(true);
    //setPalette(pal);

    int deflFrom = 0, deflTo = 0;
    m_ui->editFrom->setText(QString::number(deflFrom));
    m_ui->editTo->setText(QString::number(deflTo));
    m_ui->timeliner->setFromTo(deflFrom, deflTo);
    
    initStyleSheet();
    initSignals();
    initButtons();

    if (zeno::envconfig::get("ALWAYS"))
        m_ui->btnAlways->setChecked(true);
}

void ZTimeline::initSignals()
{
    connect(m_ui->btnPlay, SIGNAL(clicked(bool)), this, SIGNAL(playForward(bool)));
    connect(m_ui->editFrom, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->editTo, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->timeliner, SIGNAL(sliderValueChange(int)), this, SIGNAL(sliderValueChanged(int)));
    connect(m_ui->btnRun, SIGNAL(clicked()), this, SIGNAL(run()));
    connect(m_ui->btnKill, SIGNAL(clicked()), this, SIGNAL(kill()));
    connect(m_ui->btnAlways, &ZToolButton::toggled, this, [=](bool bChecked) {
        if (bChecked)
            emit alwaysChecked();
    });
    m_ui->btnAlways->setShortcut(QKeySequence(Qt::Key_F1));
    m_ui->btnRun->setShortcut(QKeySequence(Qt::Key_F2));
    m_ui->btnKill->setShortcut(QKeySequence("Shift+F2"));
    //m_ui->btnBackward->setShortcut(QKeySequence("Shift+F3"));
    //m_ui->btnForward->setShortcut(QKeySequence("F3"));
    //m_ui->btnPlay->setShortcut(QKeySequence("F4"));
    connect(m_ui->btnBackward, &ZIconLabel::clicked, this, [=]() {
        int frame = m_ui->timeliner->value();
        auto ft = fromTo();
        int frameFrom = ft.first, frameTo = ft.second;
        if (frame > frameFrom && frameFrom >= 0)
        {
            m_ui->timeliner->setSliderValue(frame - 1);
        }
    });
    connect(m_ui->btnForward, &ZIconLabel::clicked, this, [=]() {
        int frame = m_ui->timeliner->value();
        auto ft = fromTo();
        int frameFrom = ft.first, frameTo = ft.second;
        if (frame < frameTo)
        {
            m_ui->timeliner->setSliderValue(frame + 1);
        }
    });
    connect(m_ui->editFrame, &QLineEdit::editingFinished, this, [=]() {
        int frame = m_ui->editFrame->text().toInt();
        m_ui->timeliner->setSliderValue(frame);
    });
    connect(this, &ZTimeline::sliderValueChanged, this, [=]() {
        QString numText = QString::number(m_ui->timeliner->value());
        m_ui->editFrame->setText(numText);
    });
    connect(m_ui->btnAlways, &ZToolButton::toggled, [=](bool bChecked) {
        if (bChecked) {
            emit run();
        }
    });
    connect(&Zenovis::GetInstance(), SIGNAL(frameUpdated(int)), this, SLOT(onTimelineUpdate(int)));
}

void ZTimeline::initStyleSheet()
{
    auto editors = findChildren<QLineEdit *>(QString(), Qt::FindDirectChildrenOnly);
    for (QLineEdit *pLineEdit : editors) {
        pLineEdit->setProperty("cssClass", "FCurve-lineedit");
    }
}

void ZTimeline::initButtons()
{
    QSize sz = ZenoStyle::dpiScaledSize(QSize(24, 24));

    m_ui->btnBackToStart->setIcons(sz, ":/icons/start-frame.svg", ":/icons/start-frame-hover.svg");
    m_ui->btnBackward->setIcons(sz, ":/icons/previous-frame.svg", ":/icons/previous-frame-hover.svg");
    m_ui->btnPlay->setIcons(sz, ":/icons/pause.svg", ":/icons/pause-hover.svg", ":/icons/play.svg", ":/icons/play-hover.svg");
    m_ui->btnForward->setIcons(sz, ":/icons/next-frame.svg", ":/icons/next-frame-hover.svg");
    m_ui->btnForwardToEnd->setIcons(sz, ":/icons/end-frame.svg", ":/icons/end-frame-hover.svg");
    m_ui->btnRecycle->setIcons(sz, ":/icons/timeline-loop.svg", "");

    QColor bg(35, 40, 47);
    m_ui->btnAlways->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_Checkable);
    m_ui->btnAlways->setIcon(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/always-off.svg", "",
                             ":/icons/always-on.svg", "");
    m_ui->btnAlways->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnAlways->setBackgroundClr(bg, bg, bg, bg);

    m_ui->btnRun->setButtonOptions(ZToolButton::Opt_HasText);
    m_ui->btnRun->setText(tr("RUN"));
    m_ui->btnRun->setFont(QFont("Segoe UI Bold", 10));
    m_ui->btnRun->setMargins(QMargins(8, 6, 8, 6));
    m_ui->btnRun->setBackgroundClr(bg, bg, bg, bg);

    m_ui->btnKill->setButtonOptions(ZToolButton::Opt_HasText);
    m_ui->btnKill->setText(tr("Kill"));
    m_ui->btnKill->setFont(QFont("Segoe UI Bold", 10));
    m_ui->btnKill->setTextClr(QColor("#C95449"), QColor("#C95449"), QColor("#C95449"), QColor("#C95449"));
    m_ui->btnKill->setMargins(QMargins(8, 6, 8, 6));
    m_ui->btnKill->setBackgroundClr(bg, bg, bg, bg);

    m_ui->btnSound->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_Checkable);
    m_ui->btnSound->setIcon(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/sound-off.svg", "",
                             ":/icons/sound-on.svg", "");
    m_ui->btnSound->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnSound->setBackgroundClr(bg, bg, bg, bg);

    m_ui->btnSettings->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnSettings->setIcon(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/timeline-settings.svg", "",
                            ":/icons/timeline-settings.svg", "");
    m_ui->btnSettings->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnSettings->setBackgroundClr(bg, bg, bg, bg);
}

void ZTimeline::initSize()
{

}

void ZTimeline::onTimelineUpdate(int frameid)
{
    bool blocked = m_ui->timeliner->signalsBlocked();
    BlockSignalScope scope(m_ui->timeliner);
    BlockSignalScope scope2(m_ui->editFrame);
    m_ui->timeliner->setSliderValue(frameid);
    m_ui->editFrame->setText(QString::number(frameid));
}

void ZTimeline::setSliderValue(int frameid)
{
    m_ui->timeliner->setSliderValue(frameid);
}

void ZTimeline::setPlayButtonToggle(bool bToggle)
{
    m_ui->btnPlay->toggle(bToggle);
}

void ZTimeline::onFrameEditted()
{
    if (m_ui->editFrom->text().isEmpty() || m_ui->editTo->text().isEmpty())
        return;
    int frameFrom = m_ui->editFrom->text().toInt();
    int frameTo = m_ui->editTo->text().toInt();
    if (frameTo >= frameFrom)
        m_ui->timeliner->setFromTo(frameFrom, frameTo);
}

QPair<int, int> ZTimeline::fromTo() const
{
    bool bOk = false;
    int frameFrom = m_ui->editFrom->text().toInt(&bOk);
    int frameTo = m_ui->editTo->text().toInt(&bOk);
    return { frameFrom, frameTo };
}

bool ZTimeline::isAlways() const
{
    return m_ui->btnAlways->isChecked();
}

void ZTimeline::setFromTo(int frameFrom, int frameTo)
{
    m_ui->editFrame->setText(QString::number(frameFrom));
    m_ui->editTo->setText(QString::number(frameTo));
    if (frameTo >= frameFrom)
        m_ui->timeliner->setFromTo(frameFrom, frameTo);
}

void ZTimeline::setAlways(bool bOn)
{
    m_ui->btnAlways->setChecked(bOn);
}

void ZTimeline::resetSlider()
{
    m_ui->timeliner->setSliderValue(0);
}

int ZTimeline::value() const
{
    return m_ui->timeliner->value();
}
