#ifndef __ZFORKSUBGRAPHDLG_H__
#define __ZFORKSUBGRAPHDLG_H__

#include <QtWidgets>
#include "zenoui/comctrl/dialog/zframelessdialog.h"
#include <rapidjson/document.h>

class ZForkSubgraphDlg : public ZFramelessDialog
{
    Q_OBJECT
public:
    ZForkSubgraphDlg(const QMap<QString, QString>   & subgs, QWidget* parent = nullptr);
signals:
private slots:
    void onOkClicked();
    void onImportClicked();
private:
    void initUi();
private:
    QString m_version;
    QTableWidget* m_pTableWidget;
    QMap<QString, QString> m_subgsMap;
    QString m_importPath;
    QPushButton* m_pImportBtn;
};

#endif