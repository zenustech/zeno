#pragma once

#include <QtWidgets>
#include <QAbstractItemModel>

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
    void setupModelData();

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
};

class zenooutline : public QWidget
{
    Q_OBJECT

public:
    zenooutline(QWidget *parent = nullptr);
    ~zenooutline();

private:
    void setupTreeView();
    
    QTreeView *m_treeView = nullptr;
    OutlineItemModel *m_model = nullptr;
};
