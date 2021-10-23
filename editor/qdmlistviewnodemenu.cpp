#include "qdmlistviewnodemenu.h"
#include <zeno/dop/Descriptor.h>

QDMListViewNodeMenu::QDMListViewNodeMenu(QWidget *parent)
    : QListView(parent)
    , model(new QStandardItemModel(this))
{
    for (auto const &[k, d]: zeno::dop::desc_table()) {
        auto item = new QStandardItem();
        item->setText(QString::fromStdString(k));
        item->setEditable(false);
        items.push_back(item);
        model->appendRow(item);
    }

    connect(this, SIGNAL(clicked(QModelIndex)), this, SLOT(itemClicked(QModelIndex)));

    setModel(model);
}

QDMListViewNodeMenu::~QDMListViewNodeMenu()
{
    for (auto p: items)
        delete p;
    delete model;
}

void QDMListViewNodeMenu::itemClicked(QModelIndex index)
{
    auto item = model->item(index.row());
    qInfo() << "itemClicked:" << item->text();
}
