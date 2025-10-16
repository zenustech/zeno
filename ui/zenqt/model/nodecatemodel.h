#pragma once

#ifndef __NODE_CATE_MODEL_H__
#define __NODE_CATE_MODEL_H__

#include <QObject>
#include <QAbstractListModel>
#include <QString>
#include <QQuickItem>

class GraphModel;

class QmlNodeCateRole
{
    Q_GADGET
public:
    explicit QmlNodeCateRole() {}

    enum Value {    //对应common.h的NodeDataGroup
        Name,       //category name or nodename
        CateNodes,
        IsCategory,
        Keywords,
        Category,   //搜索节点时对应的Category名字，如果是Category菜单，用Name就行
        LastItem,   //是否为搜索项的最后一项，用于提示UI做些事情
    };
    Q_ENUM(Value)
};

enum NewNodeCase
{
    Case_Normal,
    Case_Foreach_Count,
    Case_Foreach_PrimAttr
};

class MenuEventFilter : public QObject
{
    Q_OBJECT
    QML_ELEMENT

public:
    explicit MenuEventFilter(QObject* parent = nullptr) : QObject(parent) {}

    Q_INVOKABLE void listenTo(QObject* object)
    {
        if (!object)
            return;

        QObjectList children = object->children();
        for (auto pObject : children) {
            pObject->installEventFilter(this);
        }
    }

signals:
    void textAppended(QString);
    void textRemoved();

protected:
    bool eventFilter(QObject* obj, QEvent* event) override {
        if (event->type() == QEvent::KeyPress) {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            int key = keyEvent->key();
            QChar c(key);
            if (c.isLetterOrNumber())
            {
                emit textAppended(c);
                return true;
            }
            if (c == Qt::Key_Backspace)
            {
                emit textRemoved();
                return true;
            }
        }
        return QObject::eventFilter(obj, event);
    }
};

class NodeCateModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;
    QML_ELEMENT

    struct MenuOrItem
    {
        QString name;
        QString category;
        QStringList nodes;
        QVariantList matchIndices;
        NewNodeCase newcase = Case_Normal;
        bool iscate = true;
        bool islastitem = false;
    };

public:
    NodeCateModel(QObject* parent = nullptr);
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void init();
    Q_INVOKABLE void search(const QString& name);
    Q_INVOKABLE bool execute(GraphModel* pGraphM, const QString& name, const QPoint& pt);
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    void clear();

public slots:
    void reload();

private:
    void initNewNodeCase();
    QVector<MenuOrItem> getHistoryNodes();

    QVector<MenuOrItem> m_items;
    QVector<MenuOrItem> m_cache_cates;   //TODO: 如果ASSET发送增删，要同步到这里
    QString m_search;

    //internal:
    QMap<QString, QString> m_nodeToCate;
    QList<QString> m_condidates;
};

#endif