#ifndef __ZLOGPANEL_H__
#define __ZLOGPANEL_H__

namespace Ui {
    class LogPanel;
}

#include <QtWidgets>

class LogItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
    typedef QStyledItemDelegate _base;
public:
    LogItemDelegate(QObject *parent = nullptr);
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

private:
    QAbstractItemView *m_view;
};

class ZlogPanel : public QWidget
{
    Q_OBJECT
public:
    ZlogPanel(QWidget* parent = nullptr);

private:
    Ui::LogPanel* m_ui;
};


#endif