#include "FloatSlider.h"
#include "widgets/zlineedit.h"
#include <QHBoxLayout>
#include <QIntValidator>
#include <QDoubleValidator>
#include <cmath>
#include <QDebug>


FloatSlider::FloatSlider(bool bIntegerMode, QWidget* parent)
    : QWidget(parent)
    , m_edit(new ZLineEdit(this))
    , m_slider(new QSlider(Qt::Horizontal, this))
    , m_bInteger(bIntegerMode)
{
    m_edit->setFixedHeight(24);
    m_edit->setProperty("cssClass", "zeno2_2_lineedit");

    updateValidator();
    m_edit->setFixedWidth(70);

    auto layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_edit);
    layout->addWidget(m_slider, 1);

    connect(m_slider, &QSlider::valueChanged, this, &FloatSlider::onSliderChanged);
    connect(m_edit, &QLineEdit::editingFinished, this, &FloatSlider::onEditFinished);

    m_slider->installEventFilter(this);

    updateInternalRange();
    setFloatValue(0.0f);
}

void FloatSlider::updateValidator() {
    if (m_bInteger)
        m_edit->setValidator(new QIntValidator(this));
    else
        m_edit->setValidator(new QDoubleValidator(this));
}

void FloatSlider::updateInternalRange() {
    int steps = std::max(1, int(std::round((m_max - m_min) / m_step)));
    m_slider->setRange(0, steps);
}

float FloatSlider::floatValue() const {
    bool ok;
    float val = m_edit->text().toFloat(&ok);
    if (ok)
        return val;
    return m_min; // fallback
}

bool FloatSlider::eventFilter(QObject* watched, QEvent* event) {
    if (watched == m_slider && event->type() == QEvent::Wheel) {
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void FloatSlider::setFloatValue(float val) {
    updateSliderFromValue(val, false);
    updateEditBox(val);
}

void FloatSlider::setFloatMinimum(float min) {
    m_min = min;
    updateInternalRange();
}

void FloatSlider::setFloatMaximum(float max) {
    m_max = max;
    updateInternalRange();
}

void FloatSlider::setFloatStep(float step) {
    if (step <= 0) return;
    m_step = step;
    updateInternalRange();
}

void FloatSlider::updateEditBox(float value) {
    if (m_bInteger)
        m_edit->setText(QString::number(int(std::round(value)), 10));
    else
        m_edit->setText(QString::number(value, 'f', 4));
}

void FloatSlider::updateSliderFromValue(float val, bool emitSignal) {
    float clamped = std::clamp(val, m_min, m_max);
    float range = m_max - m_min;
    if (range <= 0) return;

    int steps = m_slider->maximum();
    int newInt = int(std::round((clamped - m_min) / range * steps));

    if (m_slider->value() != newInt) {
        if (!emitSignal)
            m_slider->blockSignals(true);
        m_slider->setValue(newInt);
        if (!emitSignal)
            m_slider->blockSignals(false);
    }
}

void FloatSlider::onSliderChanged(int) {
    if (m_syncing) return;
    m_syncing = true;

    m_freeMode = false;
    float newValue = m_min + (float(m_slider->value()) / m_slider->maximum()) * (m_max - m_min);
    updateEditBox(newValue);
    emit floatValueChanged(newValue);

    m_syncing = false;
}

void FloatSlider::onEditFinished() {
    if (m_syncing) return;
    m_syncing = true;

    bool ok = false;
    float val = m_edit->text().toFloat(&ok);
    if (!ok) {
        m_syncing = false;
        return;
    }

    float min = m_min;
    float max = m_max;
    float step = m_step;

    float stepIndex = (val - min) / step;
    float nearest = std::round(stepIndex);
    float snappedVal = min + nearest * step;

    if (std::fabs(stepIndex - nearest) < 1e-6f) {
        updateSliderFromValue(snappedVal);
        m_freeMode = false;
        emit floatValueChanged(val);
    }
    else {
        if (!m_allowOutOfRange) {
            if (val < min) val = min;
            if (val > max) val = max;
        }
        if (val < min)
            updateSliderFromValue(min);
        else if (val > max)
            updateSliderFromValue(max);
        else
            updateSliderFromValue(val);

        m_freeMode = true;
        emit floatValueChanged(val);
    }

    m_syncing = false;
}
