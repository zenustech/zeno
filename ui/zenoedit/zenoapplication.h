#ifndef __ZENO_APPLICATION_H__
#define __ZENO_APPLICATION_H__

#include <QtWidgets>
#include "zwidgetostream.h"
#include <zeno/utils/scope_exit.h>

class GraphsManagment;
class ZenoMainWindow;
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
class ZTcpServer;
#endif
class GraphsModel;

class ZenoApplication : public QApplication
{
	Q_OBJECT
public:
    ZenoApplication(int &argc, char **argv);
    ~ZenoApplication();
    GraphsManagment *graphsManagment() const;
    void initFonts();
    void initStyleSheets();
    void setIOProcessing(bool bIOProcessing);
    bool IsIOProcessing() const;
    ZenoMainWindow* getMainWindow();
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    ZTcpServer* getServer();
#endif
    QStandardItemModel* logModel() const;

private:
    QString readQss(const QString& qssPath);

    GraphsManagment *m_pGraphs;
    ZWidgetErrStream m_errSteam;
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    ZTcpServer* m_server;
#endif
    bool m_bIOProcessing;
    QDir m_appDataPath;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#define DlgInEventLoopScope                                                             \
    zeno::scope_exit sp([=]() { zenoApp->getMainWindow()->setInDlgEventLoop(false); }); \
    zenoApp->getMainWindow()->setInDlgEventLoop(true);

#endif
