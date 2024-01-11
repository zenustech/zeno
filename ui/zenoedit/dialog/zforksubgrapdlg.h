#ifndef __ZFORKSUBGRAPHDLG_H__
#define __ZFORKSUBGRAPHDLG_H__

#include <QtWidgets>
#include "zenoui/comctrl/dialog/zframelessdialog.h"
#include <rapidjson/document.h>
#include "zmathpresetsubgrapdlg.h"

class ZForkSubgraphDlg : public ZFramelessDialog
{
    Q_OBJECT
public:
    ZForkSubgraphDlg(const QMap<QString, QString>   & subgs, QWidget* parent = nullptr);
    void setPos(const QPointF& pos);
signals:
private slots:
    void onOkClicked();
    void onImportClicked();
private:
    void initUi();
    QMap<QString, QMap<QString, QVariant>> readFile();
private:
    QTableWidget* m_pTableWidget;
    QMap<QString, QString> m_subgsMap; // <mtlid, preset mat>
    QString m_importPath;
    QPushButton* m_pImportBtn;
    QMap<QString, STMatchMatInfo> m_matchInfo;
    QPointF m_pos;
};

#endif