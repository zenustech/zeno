#include "qdmtreeviewgraphs.h"
#include <QStandardItemModel>

ZENO_NAMESPACE_BEGIN

QDMTreeViewGraphs::QDMTreeViewGraphs(QWidget *parent)
    : QTreeView(parent)
{
    auto model = new QStandardItemModel(this);

    connect(this, &QTreeView::clicked, [=, this] (QModelIndex index) {
        auto item = model->item(index.row());
        emit entryClicked(item->text());
    });

    setModel(model);
}

QDMTreeViewGraphs::~QDMTreeViewGraphs() = default;

void QDMTreeViewGraphs::setRootScene(QDMGraphicsScene *scene)
{
    rootScene = scene;

    auto touch = [&] (auto &&touch, auto *parItem, std::vector<QDMGraphicsScene *> const &scenes) -> void {
        for (auto const &scene: scenes) {
            auto const &name = scene->getName();
            auto item = new QStandardItem;
            item->setText(QString::fromStdString(name.empty() ? "(unnamed)" : name));
            item->setEditable(false);
            raiiItems.emplace_back(item);
            touch(touch, item, scene->getChildScenes());
            parItem->appendRow(item);
        }
    };
    touch(touch, static_cast<QStandardItemModel *>(model()), {rootScene});
}

ZENO_NAMESPACE_END
