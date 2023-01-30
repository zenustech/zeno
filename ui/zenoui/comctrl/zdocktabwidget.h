#ifndef __ZTABWIDGET_H__
#define __ZTABWIDGET_H__

#include <QtWidgets>

class ZDockTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    explicit ZDockTabWidget(QWidget* parent = nullptr);
    ~ZDockTabWidget();

protected:
    void paintEvent(QPaintEvent* e) override;
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

signals:
    void addClicked();
    void layoutBtnClicked();
    void tabClosed(int i);

private:
    void initStyleSheet();
};

#endif