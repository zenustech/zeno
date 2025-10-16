#pragma once
#include <QWidget>
#include <QSlider>
#include <QWheelEvent>

class ZLineEdit;

class FloatSlider : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(float value READ floatValue WRITE setFloatValue NOTIFY floatValueChanged)
    Q_PROPERTY(float minimum READ floatMinimum WRITE setFloatMinimum)
    Q_PROPERTY(float maximum READ floatMaximum WRITE setFloatMaximum)
    Q_PROPERTY(float step READ floatStep WRITE setFloatStep)
    Q_PROPERTY(bool allowOutOfRange READ allowOutOfRange WRITE setAllowOutOfRange)

public:
    explicit FloatSlider(bool bIntegerMode, QWidget* parent = nullptr);

    float floatValue() const;
    void setFloatValue(float value);

    float floatMinimum() const { return m_min; }
    void setFloatMinimum(float min);

    float floatMaximum() const { return m_max; }
    void setFloatMaximum(float max);

    float floatStep() const { return m_step; }
    void setFloatStep(float step);

    bool allowOutOfRange() const { return m_allowOutOfRange; }
    void setAllowOutOfRange(bool allow) { m_allowOutOfRange = allow; }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

signals:
    void floatValueChanged(float value);

private slots:
    void onSliderChanged(int value);
    void onEditFinished();

private:
    void updateInternalRange();
    void updateEditBox(float value);
    void updateSliderFromValue(float val, bool emitSignal = true);
    void updateValidator();

    ZLineEdit* m_edit;
    QSlider* m_slider;

    float m_min = 0.0f;
    float m_max = 1.0f;
    float m_step = 0.1f;
    bool m_allowOutOfRange = false;
    const bool m_bInteger;

    bool m_syncing = false;
    bool m_freeMode = false;
};
