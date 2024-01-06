#ifndef __ZFRAMELESSDIALOG_H__
#define __ZFRAMELESSDIALOG_H__

#include <QtWidgets>

class ZFramelessDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ZFramelessDialog(QWidget* parent = nullptr);
    ~ZFramelessDialog();

    void setMainWidget(QWidget* pWidget);
    void setTitleIcon(const QIcon& icon);
    void setTitleText(const QString& text);

protected:
    void mousePressEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void keyPressEvent(QKeyEvent* e) override;

private:
    void initTitleWidget();

    QLabel* m_pLbIcon;
    QLabel* m_pLbTitle;
    QPoint m_movePos;
};
#endif

