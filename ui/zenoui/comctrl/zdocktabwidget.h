#ifndef __ZTABWIDGET_H__
#define __ZTABWIDGET_H__

#include <QtWidgets>

class ZDockTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    explicit ZDockTabWidget(QWidget* parent = nullptr);
    ~ZDockTabWidget();

    int addTab(QWidget* widget, const QString&);
    int addTab(QWidget* widget, const QIcon& icon, const QString& label);

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

private:
    void initStyleSheet();
    QRect buttonRect();

    bool m_bHovered;
};

#endif