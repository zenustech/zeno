#ifndef __ZENO_SUBGRAPH_MODEL_H__
#define __ZENO_SUBGRAPH_MODEL_H__

#include <QModelIndex>
#include <QString>
#include <QObject>
#include <memory>

struct PlainNodeItem
{
    void setData(const QVariant &value, int role) {
        m_datas[role] = value;
    }

    QVariant data(int role) const {
        auto it = m_datas.find(role);
        if (it == m_datas.end())
            return QVariant();
        return it->second;
    }

    std::map<int, QVariant> m_datas;
};

typedef std::shared_ptr<PlainNodeItem> NODEITEM_PTR;

class SubGraphModel : public QAbstractItemModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;
public:
	explicit SubGraphModel(QObject* parent = nullptr);
	~SubGraphModel();

	//QAbstractItemModel
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
	QModelIndex parent(const QModelIndex& child) const override;
	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    bool hasChildren(const QModelIndex &parent = QModelIndex()) const override;
	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
	QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const override;
    bool setHeaderData(int section, Qt::Orientation orientation, const QVariant &value,
                       int role = Qt::EditRole) override;
    QMap<int, QVariant> itemData(const QModelIndex &index) const override;
    bool setItemData(const QModelIndex &index, const QMap<int, QVariant> &roles) override;
	QModelIndexList match(const QModelIndex &start, int role,
                          const QVariant &value, int hits = 1,
                          Qt::MatchFlags flags =
                          Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    QHash<int, QByteArray> roleNames() const override;
    bool insertRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;

	//SubGraphModel
    QModelIndex index(QString id, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex indexFromItem(PlainNodeItem* pItem) const;
    void appendItem(NODEITEM_PTR pItem);
    void removeNode(const QModelIndex& index);
    void removeLink(const QString& outputId, const QString& outputPort,
                const QString& inputId, const QString& inputPort);
    void addLink(const QString& outNode, const QString& outSock,
                const QString& inNode, const QString& inSock);
    bool insertRow(int row, NODEITEM_PTR pItem, const QModelIndex &parent = QModelIndex());
    void setName(const QString& name);
    QString name() const;

signals:
    void linkChanged(bool bAdd, const QString& outputId, const QString& outputPort,
                const QString& inputId, const QString& inputPort);

private:
    PlainNodeItem* itemFromIndex(const QModelIndex &index) const;
    bool removeRow(int row, const QModelIndex &parent = QModelIndex());

    QString m_name;
    std::map<QString, int> m_key2Row;
    std::map<int, QString> m_row2Key;
    std::unordered_map<QString, QString> m_name2Id;
    std::unordered_map<QString, NODEITEM_PTR> m_nodes;
};

//Q_DECLARE_METATYPE(SubGraphModel*)

#endif
