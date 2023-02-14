#include "ztimeline.h"
#include "zslider.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/viewportwidget.h"
#include <comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/effect/innershadoweffect.h>
#include <zeno/utils/envconfig.h>
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/zwidgetfactory.h>
#include "../viewport/zenovis.h"
#include "ui_ztimeline.h"
#include <zenoui/comctrl/view/zcomboboxitemdelegate.h>
#include "viewport/zenovis.h"
#include <zenovis/DrawOptions.h>
#include <iostream>

//////////////////////////////////////////////
ZTimeline::ZTimeline(QWidget* parent)
    : QWidget(parent)
{
    m_ui = new Ui::Timeline;
    m_ui->setupUi(this);

    QStringList items = { "23.5 fps", "24 fps", "25 fps", "30 fps", "60 fps" };
    m_ui->comboBox->addItems(items);
    m_ui->comboBox->setItemDelegate(new ZComboBoxItemDelegate2(m_ui->comboBox));
    m_ui->comboBox->setFixedWidth(ZenoStyle::dpiScaled(110));
    m_ui->comboBox->setProperty("cssClass", "newstyle");
 
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
    connect(m_ui->btnPlay, SIGNAL(toggled(bool)), this, SIGNAL(playForward(bool)));
    connect(m_ui->editFrom, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->editTo, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->timeliner, SIGNAL(sliderValueChange(int)), this, SIGNAL(sliderValueChanged(int)));
    connect(m_ui->btnRun, SIGNAL(clicked()), this, SIGNAL(run()));
    connect(m_ui->btnKill, SIGNAL(clicked()), this, SIGNAL(kill()));
    connect(m_ui->btnAlways, &ZToolButton::toggled, this, [=](bool bChecked) {
        if (bChecked)
            emit alwaysChecked();
    });
    connect(m_ui->btnSimpleRender, &QPushButton::clicked, this, [=](bool bChecked) {
        //std::cout << "SR: SimpleRender " << std::boolalpha << bChecked << "\n";
        auto viewport = zenoApp->getMainWindow()->getDisplayWidget()->getViewportWidget();
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        viewport->simpleRenderChecked = bChecked;
        scene->drawOptions->simpleRender = bChecked;
        scene->drawOptions->needRefresh = true;
    });
    m_ui->btnAlways->setShortcut(QKeySequence("F1"));

    //m_ui->btnBackward->setShortcut(QKeySequence("Shift+F3"));
    //m_ui->btnForward->setShortcut(QKeySequence("F3"));
    //m_ui->btnPlay->setShortcut(QKeySequence("F4"));
    connect(m_ui->btnBackward, &ZToolButton::clicked, this, [=]() {
        int frame = m_ui->timeliner->value();
        auto ft = fromTo();
        int frameFrom = ft.first, frameTo = ft.second;
        if (frame > frameFrom && frameFrom >= 0)
        {
            m_ui->timeliner->setSliderValue(frame - 1);
        }
    });
    connect(m_ui->btnForward, &ZToolButton::clicked, this, [=]() {
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

    QColor hoverBg("#4F5963");

    m_ui->btnBackToStart->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnBackToStart->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_startFrame_idle.svg",
        ":/icons/timeline_startFrame_light.svg",
        "",
        "");
    m_ui->btnBackToStart->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnBackToStart->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);

    m_ui->btnBackward->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnBackward->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_previousFrame_idle.svg",
        ":/icons/timeline_previousFrame_light.svg",
        "",
        "");
    m_ui->btnBackward->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnBackward->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);

    m_ui->btnPlay->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_Checkable);
    m_ui->btnPlay->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_pause_idle.svg",
        ":/icons/timeline_pause_light.svg",
        ":/icons/timeline_play_idle.svg",
        ":/icons/timeline_play_light.svg");
    m_ui->btnPlay->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnPlay->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);

    m_ui->btnForward->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnForward->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_nextFrame_idle.svg",
        ":/icons/timeline_nextFrame_light.svg",
        "",
        "");
    m_ui->btnForward->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnForward->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);

    m_ui->btnForwardToEnd->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnForwardToEnd->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_endFrame_idle.svg",
        ":/icons/timeline_endFrame_light.svg",
        "",
        "");
    m_ui->btnForwardToEnd->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnForwardToEnd->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);

    m_ui->btnRecycle->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnRecycle->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_loopMethod_loop.svg",
        ":/icons/timeline_loopMethod_loop.svg",
        "",
        "");
    m_ui->btnRecycle->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnRecycle->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);

    QColor bg(35, 40, 47);
    m_ui->btnAlways->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_Checkable);
    m_ui->btnAlways->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/always-off.svg",
        "",
        ":/icons/always-on.svg",
        "");
    m_ui->btnAlways->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnAlways->setBackgroundClr(bg, hoverBg, bg, hoverBg);

    QFont font("Alibaba PuHuiTi", 10);
    font.setWeight(QFont::DemiBold);

    //m_ui->btnSimpleRender->setProperty("cssClass", "grayButton");
    m_ui->btnSimpleRender->setFont(font);

    m_ui->btnRun->setFont(font);
    m_ui->btnRun->setText(tr("RUN"));
    m_ui->btnRun->setShortcut(QKeySequence("F2"));

    m_ui->btnKill->setFont(font);
    m_ui->btnKill->setText(tr("Kill"));
    m_ui->btnKill->setShortcut(QKeySequence("Shift+F2"));

    m_ui->btnSound->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_Checkable);
    m_ui->btnSound->setIcon(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/sound-off.svg", "",
                             ":/icons/sound-on.svg", "");
    m_ui->btnSound->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnSound->setBackgroundClr(bg, hoverBg, bg, hoverBg);

    m_ui->btnSettings->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnSettings->setIcon(ZenoStyle::dpiScaledSize(QSize(24, 24)), ":/icons/timeline-settings.svg", "",
                            ":/icons/timeline-settings.svg", "");
    m_ui->btnSettings->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnSettings->setBackgroundClr(bg, hoverBg, bg, hoverBg);
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
    m_ui->btnPlay->setChecked(bToggle);
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

void ZTimeline::initFromTo(int frameFrom, int frameTo)
{
    BlockSignalScope s1(m_ui->timeliner);
    BlockSignalScope s2(m_ui->editFrom);
    BlockSignalScope s3(m_ui->editTo);

    m_ui->editFrom->setText(QString::number(frameFrom));
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
