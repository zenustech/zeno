#ifndef __ZLINEEDIT_H__
#define __ZLINEEDIT_H__

#include <QtWidgets>

class ZScaleSlider;

class ZLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    explicit ZLineEdit(QWidget* parent = nullptr);
    explicit ZLineEdit(const QString& text, QWidget* parent = nullptr);
    void setScalesSlider(QVector<qreal> scales);

protected:
    bool event(QEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event);
    void keyPressEvent(QKeyEvent* event);
    void keyReleaseEvent(QKeyEvent* event);

private:
    void popup();

    QVector<qreal> m_scales;
    ZScaleSlider* m_pSlider;
};

#endif