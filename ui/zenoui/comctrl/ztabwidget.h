#ifndef __ZTABWIDGET_H__
#define __ZTABWIDGET_H__

#include <QtWidgets>
#include "ztabbar.h"

class ZTabWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ZTabWidget(QWidget* parent = nullptr);
    ~ZTabWidget();

    int addTab(const QString& label, const QIcon& icon);
    int addTab(const QString& label);
    void removeTab(int index);
    QString tabText(int index) const;
    void setTabText(int index, const QString& text);

public slots:
    void setCurrentIndex(int index);
    void setCurrentIndex(const QString& text);

signals:
    void currentChanged(int index);
    void tabCloseClicked(int index);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void init();
    void _removeTab(int index);

    QStackedWidget* m_stack;
    ZTabBar* m_pTabbar;
};

#endif