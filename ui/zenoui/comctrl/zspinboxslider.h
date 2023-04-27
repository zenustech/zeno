#ifndef __ZSPINBOX_SLIDER_H__
#define __ZSPINBOX_SLIDER_H__

#include <QtWidgets>

class ZSpinBoxSlider : public QWidget
{
    Q_OBJECT
public:
    ZSpinBoxSlider(QWidget* parent = nullptr);
    void setValue(int value);
    void setRange(int from, int to);
    void setSingleStep(int step);
    int value() const;

signals:
    void valueChanged(int);

private slots:
    void onValueChanged(int);

private:
    QSpinBox* m_pSpinbox;
    QSlider* m_pSlider;
};

#endif