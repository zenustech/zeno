#ifndef __ZENO_APPLICATION_H__
#define __ZENO_APPLICATION_H__

#include <QtWidgets>
#include "zwidgetostream.h"
#include <zeno/utils/scope_exit.h>

class GraphsManagment;
class ZenoMainWindow;
#ifdef ZENO_MULTIPROCESS
class ZTcpServer;
#endif
class GraphsModel;

class ZenoApplication : public QApplication
{
	Q_OBJECT
public:
    ZenoApplication(int &argc, char **argv);
    ~ZenoApplication();
    QSharedPointer<GraphsManagment> graphsManagment() const;
    void initFonts();
    void initStyleSheets();
    void setIOProcessing(bool bIOProcessing);
    bool IsIOProcessing() const;
    ZenoMainWindow* getMainWindow();
#ifdef ZENO_MULTIPROCESS
    ZTcpServer* getServer();
#endif
    QStandardItemModel* logModel() const;

private:
    QString readQss(const QString& qssPath);

    QSharedPointer<GraphsManagment> m_pGraphs;
    ZWidgetErrStream m_errSteam;
#ifdef ZENO_MULTIPROCESS
    ZTcpServer* m_server;
#endif
    bool m_bIOProcessing;
    QDir m_appDataPath;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#endif
