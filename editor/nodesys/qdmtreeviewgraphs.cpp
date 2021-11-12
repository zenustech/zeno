#include "qdmtreeviewgraphs.h"
#include <zeno/dop/Descriptor.h>
#include <QStandardItemModel>
#include <QStandardItem>

ZENO_NAMESPACE_BEGIN

QDMTreeViewGraphs::QDMTreeViewGraphs(QWidget *parent)
    : QTreeView(parent)
{
    auto model = new QStandardItemModel(this);
    for (auto const &k: std::to_array<std::string>({"main", "sub"})) {
        auto item = new QStandardItem;
        item->setText(QString::fromStdString(k));
        item->setEditable(false);
        items.emplace_back(item);
        model->appendRow(item);
    }

    connect(this, &QTreeView::clicked, [=, this] (QModelIndex index) {
        auto item = model->item(index.row());
        emit entryClicked(item->text());
    });

    setModel(model);
}

QDMTreeViewGraphs::~QDMTreeViewGraphs() = default;

ZENO_NAMESPACE_END
