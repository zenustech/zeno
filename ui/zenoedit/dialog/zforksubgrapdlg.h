#ifndef __ZFORKSUBGRAPHDLG_H__
#define __ZFORKSUBGRAPHDLG_H__

#include <QtWidgets>
#include "zenoui/comctrl/dialog/zframelessdialog.h"

class ZForkSubgraphDlg : public ZFramelessDialog
{
    Q_OBJECT
public:
    ZForkSubgraphDlg(const QMap<QString, QString>   & subgs, QWidget* parent = nullptr);
signals:

private:
    void initUi();
private:
    QString m_version;
    QTableWidget* m_pTableWidget;
    QMap<QString, QString> m_subgsMap;
};

#endif