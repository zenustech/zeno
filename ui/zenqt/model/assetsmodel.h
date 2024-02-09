#ifndef __ASSETSMODEL_H__
#define __ASSETSMODEL_H__

#include <QtWidgets>
#include <zeno/core/data.h>
#include <QQuickItem>
#include "uicommon.h"
#include <zeno/core/Assets.h>

class GraphModel;

class AssetsModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;
    QML_ELEMENT

    struct _AssetItem
    {
        zeno::AssetInfo info;
        GraphModel* pGraphM;
    };

public:
    AssetsModel(QObject* parent = nullptr);
    ~AssetsModel();

    void init(const zeno::AssetsData& assets);
    void clear();
    Q_INVOKABLE GraphModel* getAssetGraph(const QString& graphName) const;

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QModelIndexList match(const QModelIndex& start, int role,
        const QVariant& value, int hits = 1,
        Qt::MatchFlags flags =
        Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    QHash<int, QByteArray> roleNames() const override;

    void newAsset(const zeno::AssetInfo asset);
    void addAsset(const zeno::GraphData& graph);
    void removeAsset(const QString& assetName);
    zeno::AssetInfo getAsset(const QString& assetName) const;

private:
    void _addAsset(zeno::AssetInfo info);
    void _removeAsset(const QString& newName);
    int rowByName(const QString& name) const;

    QVector<_AssetItem> m_assets;

    std::string m_cbCreateAsset;
    std::string m_cbRemoveAsset;
    std::string m_cbRenameAsset;
};

#endif