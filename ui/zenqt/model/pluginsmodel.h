#pragma once

#ifndef __PLUGINS_MODEL_H__
#define __PLUGINS_MODEL_H__

#include <QtWidgets>
#include <QQuickItem>

class PluginsModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;
    QML_ELEMENT

    struct _pluginItem
    {
        QString path;   //插件dll的路径
#ifdef _WIN32
        HMODULE hDll;
#else
        
#endif
        bool bLoaded;   //false为加载不正常，比如路径不存在
    };
public:
    PluginsModel(QObject* parent = nullptr);
    ~PluginsModel();

    Q_INVOKABLE void removePlugin(int row);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QModelIndexList match(const QModelIndex& start, int role,
        const QVariant& value, int hits = 1,
        Qt::MatchFlags flags =
        Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    QHash<int, QByteArray> roleNames() const override;

    void addPlugin(const QString& path);

private:
    QVector<_pluginItem> m_items;
};


#endif