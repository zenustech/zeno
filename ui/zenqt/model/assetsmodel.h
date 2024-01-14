#ifndef __ASSETSMODEL_H__
#define __ASSETSMODEL_H__

#include <QtWidgets>
#include <zeno/core/data.h>
#include <QQuickItem>

class GraphModel;

class AssetsModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;
    QML_ELEMENT

public:
    AssetsModel(QObject* parent = nullptr);
    ~AssetsModel();

    void init(const zeno::AssetsData& assets);
    void clear();
    Q_INVOKABLE GraphModel* getAsset(const QString& graphName) const;

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QModelIndexList match(const QModelIndex& start, int role,
        const QVariant& value, int hits = 1,
        Qt::MatchFlags flags =
        Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    QHash<int, QByteArray> roleNames() const override;

    void newAsset(const QString& assetName);
    void addAsset(const zeno::GraphData& graph);
    void removeAsset(const QString& assetName);

private:
    QVector<GraphModel*> m_assets;
};

#endif