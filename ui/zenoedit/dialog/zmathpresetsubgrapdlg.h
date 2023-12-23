#ifndef __ZMatchPresetSubgraphDlg_H__
#define __ZMatchPresetSubgraphDlg_H__

#include <QtWidgets>
#include "zenoui/comctrl/dialog/zframelessdialog.h"
#include <rapidjson/document.h>

struct STMatchMatInfo {
    QString m_matType;
    QMap<QString, QString> m_matchInfo;
    QSet<QString> m_matKeys;
};

class ZMatchPresetSubgraphDlg : public ZFramelessDialog
{
    Q_OBJECT
public:
    ZMatchPresetSubgraphDlg(const QMap<QString, STMatchMatInfo>& info, QWidget* parent = nullptr);
    static QMap<QString, STMatchMatInfo> getMatchInfo(const QMap<QString, STMatchMatInfo>& info, QWidget* parent = nullptr);
signals:
private slots:
    void onRowInserted(const QModelIndex& parent, int first, int last);
private:
    void initUi(const QMap<QString, STMatchMatInfo>& info);
    void initModel();
private:
    QTreeView* m_pTreeView;
    QStandardItemModel*m_pModel;
    QMap<QString, STMatchMatInfo> m_matchInfos;
};

#endif