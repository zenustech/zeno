#include "qdmtreeviewgraphs.h"
#include <QStandardItemModel>
#include <zeno/ztd/memory.h>
#include <zeno/zmt/log.h>

ZENO_NAMESPACE_BEGIN

namespace {

struct QDMZenoSceneItem final : QStandardItem
{
    QDMGraphicsScene *scene;

    explicit QDMZenoSceneItem(QDMGraphicsScene *scene)
        : scene(scene)
    {
        setEditable(false);
    }
};

}

QDMTreeViewGraphs::QDMTreeViewGraphs(QWidget *parent)
    : QTreeView(parent)
{
    auto model = new QStandardItemModel(this);

    connect(this, &QTreeView::clicked, [=, this] (QModelIndex index) {
        std::vector<QModelIndex> indices;
        do {
            indices.push_back(index);
            index = index.parent();
        } while (index.isValid());
        std::reverse(indices.begin(), indices.end());
        auto item = static_cast<QDMZenoSceneItem *>(model->item(indices[0].row()));
        QString path = item->text();
        std::for_each(indices.begin() + 1, indices.end(), [&] (auto const &idx) {
            path += "/" + item->text();
            item = static_cast<QDMZenoSceneItem *>(item->child(idx.row()));
        });
        emit entryClicked(path);
        ZENO_LOG_DEBUG("clicked {}", index.row());
    });
    connect(this, &QTreeView::doubleClicked, [=, this] (QModelIndex index) {
        auto item = static_cast<QDMZenoSceneItem *>(model->item(index.row()));
        ZENO_LOG_DEBUG("double clicked {}", index.row());
        auto chScene = std::make_unique<QDMGraphicsScene>();
        chScene->name.set("child1");
        item->scene->childScenes.add(std::move(chScene));
        refreshRootScene();
    });

    setModel(model);
}

QDMTreeViewGraphs::~QDMTreeViewGraphs() = default;

void QDMTreeViewGraphs::refreshRootScene()
{
    setRootScene(rootScene);
}

void QDMTreeViewGraphs::setRootScene(QDMGraphicsScene *scene)
{
    while (model()->rowCount())
        model()->removeRow(0);

    rootScene = scene;

    auto touch = [this] (auto &&touch, auto *parItem, auto const &scenes) -> void {
        for (auto const &scene: scenes) {
            auto item = new QDMZenoSceneItem(ztd::get_ptr(scene));
            auto vname = scene->name.view([=] (std::string const &name) {
                item->setText(QString::fromStdString(name.empty() ? "(unnamed)" : name));
            });
#if XXX
            connect(item, QStandardItem::editingFinished, this, [=] {
                vname.set(item.text().toStdString());
            });
#endif
            touch(touch, item, scene->childScenes.get());
            parItem->appendRow(item);
        }
    };
    touch(touch,
          static_cast<QStandardItemModel *>(model()),
          std::vector<QDMGraphicsScene *>{rootScene});
}

ZENO_NAMESPACE_END
