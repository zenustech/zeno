#ifndef __NODEITEM_H__
#define __NODEITEM_H__

#include <QString>
#include <QRectF>
#include <vector>
#include <QtWidgets>
#include <zenoui/model/modelrole.h>

struct SocketItem
{
	QString name;
};

struct ParamsItem {
};

struct NodeItem;

#define ENABLE_SMART_POINTER

#ifdef ENABLE_SMART_POINTER
typedef std::shared_ptr<NodeItem> SP_NODE_ITEM;
typedef std::weak_ptr<NodeItem> WP_NODEITEM;
#else
typedef NodeItem* SP_NODE_ITEM;
typedef NodeItem* WP_NODEITEM;
#endif


struct NodeItem
{
    typedef std::unordered_map<QString, NodeItem *> MAPPER;

	NodeItem() = default;

	int childrenCount() const { return m_childrens.size(); }
    SP_NODE_ITEM child(int index) const
    {
        if (index < 0 || index >= childrenCount())
            return nullptr;

        auto itRow = m_row2Key.find(index);
        Q_ASSERT(itRow != m_row2Key.end()); 
		auto itItem = m_childrens.find(itRow->second);
        Q_ASSERT(itItem != m_childrens.end());
		return itItem->second;
	}

    SP_NODE_ITEM parent() {
#ifdef ENABLE_SMART_POINTER
        SP_NODE_ITEM parent = m_parent.lock();
#else
        SP_NODE_ITEM parent = m_parent;
#endif
        return parent;
    }

	SP_NODE_ITEM child(const QString &id)
    {
        auto it = m_childrens.find(id);
        if (it == m_childrens.end())
            return nullptr;
        return it->second;
	}

    SP_NODE_ITEM name(const QString& name)
    {
        auto it = m_name2Id.find(name);
        Q_ASSERT(it != m_name2Id.end());
        auto itItem = m_childrens.find(it->second);
        Q_ASSERT(itItem != m_childrens.end());
        return itItem->second;
    }

    int indexOfItem(SP_NODE_ITEM pItem)
    {
        if (pItem == nullptr) return -1;

        const QString& id = data(ROLE_OBJID).toString();

        auto itRow = m_key2Row.find(id);
        Q_ASSERT(itRow != m_key2Row.end());
        return itRow->second;
    }

    void appendItem(SP_NODE_ITEM pItem)
    {
        if (!pItem)
            return;
        const QString& id = pItem->data(ROLE_OBJID).toString();
        const QString& name = pItem->data(ROLE_OBJNAME).toString();
        Q_ASSERT(!id.isEmpty() && !name.isEmpty() &&
            m_childrens.find(id) == m_childrens.end());

        m_childrens.insert(std::make_pair(id, pItem));
        m_name2Id.insert(std::make_pair(name, id));
        int nRows = m_childrens.size();
        m_row2Key.insert(std::make_pair(nRows, id));
        m_key2Row.insert(std::make_pair(id, nRows));
    }

    void insertItem(int row, SP_NODE_ITEM pItem)
    {
        const QString &id = pItem->data(ROLE_OBJID).toString();
        const QString &name = pItem->data(ROLE_OBJNAME).toString();

        Q_ASSERT(!id.isEmpty() && !name.isEmpty() &&
                 m_childrens.find(id) == m_childrens.end());

        auto itRow = m_row2Key.find(row);
        Q_ASSERT(itRow != m_row2Key.end());
        int nRows = childrenCount();
        for (int r = nRows; r >= row; r--)
        {
            const QString& key = m_row2Key[r - 1];
            m_row2Key[r] = key;
            m_key2Row[key] = r;
        }

        m_childrens.insert(std::make_pair(id, pItem));
        m_row2Key[row] = id;
        m_key2Row[id] = row;
        m_name2Id[name] = id;
    }

    void removeItem(int row) {
        //todo
    }

    void setData(const QVariant& value, int role)
    {
        m_datas[role] = value;
    }

    QVariant data(int role) const
    {
        auto it = m_datas.find(role);
        if (it == m_datas.end())
            return QVariant();
        return it->second;
    }

    std::map<QString, int> m_key2Row;
    std::map<int, QString> m_row2Key;
    std::unordered_map<QString, QString> m_name2Id;
    WP_NODEITEM m_parent;
    std::unordered_map<QString, SP_NODE_ITEM> m_childrens;
    std::map<int, QVariant> m_datas;
};

struct LinkItem
{
	QString id;
	QString srcNodeId;
	QString dstNodeId;
};

#endif
