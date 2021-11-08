#include "qdmlistviewnodemenu.h"
#include <zeno/dop/Descriptor.h>
#include <QStandardItemModel>
#include <QStandardItem>

ZENO_NAMESPACE_BEGIN

QDMListViewNodeMenu::QDMListViewNodeMenu(QWidget *parent)
    : QListView(parent)
{
    auto model = new QStandardItemModel(this);
    for (auto const &[k, d]: dop::descriptor_table()) {
        auto item = new QStandardItem;
        item->setText(QString::fromStdString(k));
        item->setEditable(false);
        items.emplace_back(item);
        model->appendRow(item);
    }

    connect(this, &QListView::clicked, [=, this] (QModelIndex index) {
        auto item = model->item(index.row());
        emit entryClicked(item->text());
    });

    setModel(model);
}

QDMListViewNodeMenu::~QDMListViewNodeMenu() = default;

ZENO_NAMESPACE_END
