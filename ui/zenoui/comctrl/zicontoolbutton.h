#ifndef __ZICON_TOOLBUTTON_H__
#define __ZICON_TOOLBUTTON_H__

#include <QToolButton>

class ZIconToolButton : public QToolButton
{
    Q_OBJECT
public:
    explicit ZIconToolButton(const QString& iconIdle, const QString& iconLight, QWidget* parent = nullptr);
    ~ZIconToolButton();
    QSize sizeHint() const override;

protected:
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    QString m_iconIdle;
    QString m_iconLight;
};


#endif