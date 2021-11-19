#include "qdmtreeviewgraphs.h"
#include <QStandardItemModel>
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

static QDMZenoSceneItem *resolveIndex(QStandardItemModel *model, QModelIndex index) {
    std::vector<QModelIndex> indices;
    do {
        indices.push_back(index);
        index = index.parent();
    } while (index.isValid());
    std::reverse(indices.begin(), indices.end());
    auto item = static_cast<QDMZenoSceneItem *>(model->item(indices[0].row()));
    //QString path = item->text();
    std::for_each(indices.begin() + 1, indices.end(), [&] (QModelIndex index) {
        item = static_cast<QDMZenoSceneItem *>(item->child(index.row()));
        //path += '/' + item->text();
    });
    return item;
}

QDMTreeViewGraphs::QDMTreeViewGraphs(QWidget *parent)
    : QTreeView(parent)
{
    auto model = new QStandardItemModel(this);

    connect(this, &QTreeView::clicked, [=, this] (QModelIndex index) {
        auto item = resolveIndex(model, index);
        emit sceneSwitched(item->scene);
    });

#if 0
    connect(this, &QTreeView::doubleClicked, [=, this] (QModelIndex index) {
        auto item = resolveIndex(model, index);

        auto chName = zan::find_unique_name
            ( item->scene->childScenes.get()
            | zan::map(ztd::get_ptr)
            | zan::map(ZENO_F1(p, p->name.get()))
            , "child");

        auto chScene = std::make_unique<QDMGraphicsScene>();
        chScene->name.set(chName);
        item->scene->childScenes.add(std::move(chScene));
        refreshRootScene();
    });
#endif

    setModel(model);
}

QDMTreeViewGraphs::~QDMTreeViewGraphs() = default;

QSize QDMTreeViewGraphs::sizeHint() const
{
    return QSize(420, 800);
}

void QDMTreeViewGraphs::refreshRootScene()
{
    setRootScene(rootScene);
}

void QDMTreeViewGraphs::setRootScene(QDMGraphicsScene *scene)
{
    while (model()->rowCount())
        model()->removeRow(0);

    rootScene = scene;

    auto touch = [this] ( auto &touch
                        , auto *parItem
                        , std::vector<QDMGraphicsScene *> const &scenes
                        ) -> void {
        for (auto *scene: scenes) {
            auto item = new QDMZenoSceneItem(scene);
            auto name = scene->getName();
            item->setText(QString::fromStdString(name.empty() ? "(unnamed)" : name));
            touch(touch, item, scene->getChildScenes());
            parItem->appendRow(item);
        }
    };
    touch(touch,
          static_cast<QStandardItemModel *>(model()),
          {rootScene});

    expandAll();

    emit rootSceneChanged(rootScene);
}

void QDMTreeViewGraphs::switchScene(QDMGraphicsScene *scene)
{
    emit sceneSwitched(scene);
}

ZENO_NAMESPACE_END
