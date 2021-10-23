#ifndef QDMLISTVIEWNODEMENU_H
#define QDMLISTVIEWNODEMENU_H

#include <QListView>
#include <QStandardItemModel>
#include <QStandardItem>
#include <vector>

class QDMListViewNodeMenu : public QListView
{
    Q_OBJECT

    QStandardItemModel *model;
    std::vector<QStandardItem *> items;

public:
    explicit QDMListViewNodeMenu(QWidget *parent = nullptr);
    ~QDMListViewNodeMenu();

signals:
    void entryClicked(QString name);
};

#endif // QDMLISTVIEWNODEMENU_H
