#include "zspinboxslider.h"
#include "style/zenostyle.h"
#include <zenomodel/include/uihelper.h>
#include "zassert.h"


ZSpinBoxSlider::ZSpinBoxSlider(QWidget* parent)
    : QWidget(parent)
    , m_pSlider(nullptr)
    , m_pSpinbox(nullptr)
{
    QHBoxLayout* pLayout = new QHBoxLayout;

    m_pSpinbox = new QSpinBox;
    m_pSpinbox->setStyleSheet(ZenoStyle::dpiScaleSheet("\
                    QSpinBox {\
                        background: #191D21;\
                        height: 28px;\
                        color: #C3D2DF;\
                        font: 14px 'Segoe UI';\
                        border: none;\
                    }\
                    \
                    QSpinBox::down-button  {\
                        subcontrol-origin: margin;\
                        subcontrol-position: center left;\
                        image: url(:/icons/leftArrow.svg);\
                        background-color: #191D21;\
                        height: 24px;\
                        width: 20px;\
                    }\
                    \
                    QSpinBox::down-button:hover {\
                        image: url(:/icons/leftArrow-on.svg);\
                    }\
                    \
                    QSpinBox::up-button  {\
                        subcontrol-origin: margin;\
                        subcontrol-position: center right;\
                        image: url(:/icons/rightArrow.svg);\
                        background-color: #191D21;\
                        height: 24px;\
                        width: 20px;\
                    }\
                    \
                    QSpinBox::up-button:hover {\
                        image: url(:/icons/rightArrow-on.svg);\
                    }\
                "));
    m_pSpinbox->setAlignment(Qt::AlignCenter);
    m_pSpinbox->setFixedWidth(ZenoStyle::dpiScaled(80));

    m_pSlider = new QSlider(Qt::Horizontal);
    m_pSlider->setStyleSheet(ZenoStyle::dpiScaleSheet("\
        QSlider::groove:horizontal {\
            height: 4px;\
            background: #707D9C;\
        }\
        \
        QSlider::handle:horizontal {\
            background: #DFE2E5;\
            width: 6px;\
            margin: -8px 0;\
        }\
        QSlider::add-page:horizontal {\
            background: #191D21;\
        }\
        \
        QSlider::sub-page:horizontal {\
            background: #707D9C;\
        }\
    "));

    setValue(0);

    pLayout->addWidget(m_pSpinbox);
    pLayout->addWidget(m_pSlider);
    pLayout->setMargin(0);

    connect(m_pSlider, &QSlider::valueChanged, this, [=](int value) {
        BlockSignalScope sp(m_pSpinbox);
        m_pSpinbox->setValue(value);
        emit valueChanged(value);
    });
    connect(m_pSpinbox, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));

    setLayout(pLayout);
}

void ZSpinBoxSlider::onValueChanged(int value)
{
    BlockSignalScope sp(m_pSlider);
    m_pSlider->setValue(value);
    emit valueChanged(value);
}

void ZSpinBoxSlider::setRange(int from, int to)
{
    BlockSignalScope scope(m_pSlider);
    BlockSignalScope scope2(m_pSlider);
    m_pSlider->setRange(from, to);
    m_pSpinbox->setRange(from, to);
    setValue(from);
}

void ZSpinBoxSlider::setSingleStep(int step)
{
    BlockSignalScope scope(m_pSlider);
    BlockSignalScope scope2(m_pSlider);
    m_pSlider->setSingleStep(step);
    m_pSpinbox->setSingleStep(step);
}

void ZSpinBoxSlider::setValue(int value)
{
    BlockSignalScope scope(m_pSlider);
    BlockSignalScope scope2(m_pSlider);
    m_pSlider->setValue(value);
    m_pSpinbox->setValue(value);
}

int ZSpinBoxSlider::value() const
{
    ZASSERT_EXIT(m_pSlider->value() == m_pSpinbox->value(), 0);
    return m_pSlider->value();
}
