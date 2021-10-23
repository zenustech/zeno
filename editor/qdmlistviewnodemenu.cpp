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

    connect(this, &QListView::clicked, [=, this] (QModelIndex index) {
        auto item = model->item(index.row());
        entryClicked(item->text());
    });

    setModel(model);
}

QDMListViewNodeMenu::~QDMListViewNodeMenu()
{
    for (auto p: items)
        delete p;
    delete model;
}
