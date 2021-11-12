#include "qdmtreeviewgraphs.h"

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

    auto touch = [&] (auto &&touch, QDMGraphicsScene *scene) {
        for (auto const &chScene: scene->getChildScenes()) {
            auto name = chScene->getName();
            auto item = new QStandardItem;
            item->setText(QString::fromStdString(name));
            item->setEditable(false);
            items.emplace_back(item);
            model->appendRow(item);
        }
    };
    touch(touch, rootScene);
}

ZENO_NAMESPACE_END
