#include "ztimeline.h"
#include "zslider.h"
#include <comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/effect/innershadoweffect.h>
#include "ui_ztimeline.h"


//////////////////////////////////////////////
ZTimeline::ZTimeline(QWidget* parent)
    : QWidget(parent)
{
    m_ui = new Ui::Timeline;
    m_ui->setupUi(this);
 
    setFocusPolicy(Qt::ClickFocus);
 //   setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(42, 42, 42));
    setAutoFillBackground(true);
    setPalette(pal);

    //m_ui->timeliner->setFromTo(0, 1);
    
    initStyleSheet();
    initSignals();

 //   connect(plblForward, SIGNAL(toggled(bool)), this, SIGNAL(playForward(bool)));
}

void ZTimeline::initSignals()
{
    connect(m_ui->editFrom, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->editTo, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->timeliner, SIGNAL(sliderValueChange(int)), this, SIGNAL(sliderValueChanged(int)));
    connect(m_ui->btnRun, &QPushButton::clicked, this, [=]() {
        if (m_ui->editFrom->text().isEmpty() || m_ui->editTo->text().isEmpty())
            return;
        int frameFrom = m_ui->editFrom->text().toInt();
        int frameTo = m_ui->editTo->text().toInt();
        if (frameTo > frameFrom)
            emit run(frameFrom, frameTo);
    });
}

void ZTimeline::initStyleSheet()
{
    auto editors = findChildren<QLineEdit *>(QString(), Qt::FindDirectChildrenOnly);
    for (QLineEdit *pLineEdit : editors) {
        pLineEdit->setProperty("cssClass", "FCurve-lineedit");
    }

    auto buttons = findChildren<QPushButton *>(QString(), Qt::FindDirectChildrenOnly);
    for (QPushButton* btn : buttons)
    {
        if (btn != m_ui->btnRun && btn != m_ui->btnKill)
            btn->setProperty("cssClass", "timeline");

        InnerShadowEffect *effect = new InnerShadowEffect;
        btn->setGraphicsEffect(effect);
    }
}

void ZTimeline::onTimelineUpdate(int frameid)
{
    bool blocked = m_ui->timeliner->signalsBlocked();
    m_ui->timeliner->blockSignals(true);
    m_ui->timeliner->setSliderValue(frameid);
    m_ui->timeliner->blockSignals(blocked);
}

void ZTimeline::onFrameEditted()
{
    if (m_ui->editFrom->text().isEmpty() || m_ui->editTo->text().isEmpty())
        return;
    int frameFrom = m_ui->editFrom->text().toInt();
    int frameTo = m_ui->editTo->text().toInt();
    if (frameTo > frameFrom)
        m_ui->timeliner->setFromTo(frameFrom, frameTo);
}
