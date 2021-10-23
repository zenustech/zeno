#include "qdmlistviewnodemenu.h"

QDMListViewNodeMenu::QDMListViewNodeMenu(QWidget *parent)
    : QListView(parent)
    , model(new QStandardItemModel(this))
{
    auto item = new QStandardItem();
    item->setText("readobjmesh");
    item->setEditable(false);
    model->appendRow(item);

    connect(this, SIGNAL(clicked(QModelIndex)), this, SLOT(itemClicked(QModelIndex)));

    setModel(model);
}

void QDMListViewNodeMenu::itemClicked(QModelIndex index) {
    qInfo() << "itemClicked:" << index.row();
}
