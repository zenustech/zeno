#pragma once

#include <QtWidgets>
#include <QAbstractItemModel>
#include <tinygltf/json.hpp>
#include <glm/glm.hpp>
#include <optional>
#include "zeno/extra/SceneAssembler.h"

using Json = nlohmann::json;
class OutlineItemModel : public QAbstractItemModel
{
    Q_OBJECT
public:
    explicit OutlineItemModel(QObject *parent = nullptr);
    ~OutlineItemModel();

    // QAbstractItemModel接口实现
    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex &child) const override;
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    
    // 自定义数据操作
    void setupModelDataFromMessage(Json const& content);
    void clearModelData();

private:
    struct OutlineItem {
        QString name;
        OutlineItem* parent = nullptr; // 父指针保持原始指针
        int row = -1;
        std::vector<std::unique_ptr<OutlineItem>> children;
        
        OutlineItem* addChild(const QString& name) {
            children.emplace_back(std::make_unique<OutlineItem>());
            auto* child = children.back().get();
            child->name = name;
            child->parent = this;
            child->row = children.size() - 1;  // 设置行索引
            return child;
        }
    };
    
    std::unique_ptr<OutlineItem> rootItem;  // rootItem也使用unique_ptr
    void OutlineItemModel::set_child_node(Json const&json, OutlineItemModel::OutlineItem *item, std::string name);
};

class zenooutline : public QWidget
{
    Q_OBJECT

public:
    zenooutline(QWidget *parent = nullptr);
    ~zenooutline();

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    void setupTreeView();
    void sendOptixMessage(Json &msg);
    
    QTreeView *m_treeView = nullptr;
    OutlineItemModel *m_model = nullptr;
};
