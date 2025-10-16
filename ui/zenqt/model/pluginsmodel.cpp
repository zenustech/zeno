#include "pluginsmodel.h"
#include "uicommon.h"
#include "settings/zsettings.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zassert.h"
#include <zeno/core/Session.h>


PluginsModel::PluginsModel(QObject* parent)
    : QAbstractListModel(parent)
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Zeno Plugins");
    QStringList lst = settings.childKeys();
    for (int i = 0; i < lst.size(); i++)
    {
        const QString& key = lst[i];
        QString path = settings.value(key).toString();
        QFileInfo fninfo(path);
        QString appDir = zenoApp->applicationDirPath();
        QString dllName = fninfo.fileName();
        path = appDir + '/' + dllName;

        _pluginItem _item;
        _item.path = path;
        _item.bLoaded = fninfo.exists();
        if (_item.bLoaded) {
            zeno::getSession().beginLoadModule(path.toStdString());
            zeno::scope_exit sp([&]() { zeno::getSession().endLoadModule(); });
#ifdef _WIN32
            _item.hDll = LoadLibrary(path.toUtf8().data());
            ZASSERT_EXIT(_item.hDll != INVALID_HANDLE_VALUE);
#else

#endif
        }
        m_items.append(_item);
    }
}

PluginsModel::~PluginsModel()
{
    for (auto _item : m_items) {
        #ifdef _WIN32
        FreeLibrary(_item.hDll);
        #else

        #endif
    }
    m_items.clear();
}

void PluginsModel::removePlugin(int row) {
    removeRows(row, 1);
}

int PluginsModel::rowCount(const QModelIndex& parent) const {
    return m_items.size();
}

QVariant PluginsModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() >= m_items.size()) {
        return QVariant();
    }
    if (role == Qt::DisplayRole) {
        QFileInfo fileInfo(m_items[index.row()].path);
        return fileInfo.fileName();
    }
    else if (role == QtRole::ROLE_PLUGIN_PATH) {
        return m_items[index.row()].path;
    }
    else if (role == QtRole::ROLE_PLUGIN_LOADED) {
        return m_items[index.row()].bLoaded;
    }
    else {
        return QVariant();
    }
}

bool PluginsModel::setData(const QModelIndex& index, const QVariant& value, int role) {
    return false;
}

QModelIndexList PluginsModel::match(
    const QModelIndex& start,
    int role,
    const QVariant& value,
    int hits,
    Qt::MatchFlags flags) const {
    return QModelIndexList();
}

static QString getFileName(const QString& path) {
    int slashIndex = path.lastIndexOf('/');
    int backslashIndex = path.lastIndexOf('\\');
    int lastSeparator = qMax(slashIndex, backslashIndex);
    return path.mid(lastSeparator + 1);
}

bool PluginsModel::removeRows(int row, int count, const QModelIndex& parent) {
#ifdef _WIN32
    beginRemoveRows(parent, row, row);
    bool ret = FreeLibrary(m_items[row].hDll);

    //先从注册表移除
    const QString& filePath = m_items[row].path;
    QFileInfo fn(filePath);
    QString name;
    if (fn.exists()) {
        name = QFileInfo(filePath).fileName();
    }
    else {
        //已经不存在了，只能从文件路径提取名字
        name = getFileName(filePath);
    }
    
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Zeno Plugins");
    settings.remove(name);

    zeno::getSession().uninstallModule(filePath.toStdString());

    m_items.removeAt(row);
    endRemoveRows();
    return true;
#else
    return false;
#endif
}

QHash<int, QByteArray> PluginsModel::roleNames() const {
    QHash<int, QByteArray> roles;
    roles[Qt::DisplayRole] = "name";
    roles[QtRole::ROLE_PLUGIN_PATH] = "path";
    roles[QtRole::ROLE_PLUGIN_LOADED] = "isLoaded";
    return roles;
}

void PluginsModel::addPlugin(const QString& filePath) {
#ifdef _WIN32
    if (!filePath.isEmpty()) {
        HMODULE hDll = 0;
        {
            zeno::getSession().beginLoadModule(filePath.toStdString());
            zeno::scope_exit sp([&]() { zeno::getSession().endLoadModule(); });
            hDll = LoadLibrary(filePath.toUtf8().data());
            //添加到注册表
            if (hDll) {
                QString name = QFileInfo(filePath).fileName();
                QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
                settings.beginGroup("Zeno Plugins");
                settings.setValue(name, filePath);
            }
        }

        //还要考虑已经加载过的
        if (hDll) {
            int nRows = m_items.size();
            beginInsertRows(QModelIndex(), nRows, nRows);
            _pluginItem item;
            item.bLoaded = true;
            item.path = filePath;
            item.hDll = hDll;
            m_items.append(item);
            endInsertRows();
        }
        else {
            //TODO: 提示框
        }
    }
#else
    //todo: linux case
#endif
}