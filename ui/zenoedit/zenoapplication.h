#ifndef __ZENO_APPLICATION_H__
#define __ZENO_APPLICATION_H__

#include <QtWidgets>
#include "zwidgetostream.h"
#include "cache/zcachemgr.h"
#include "uilogic/procclipboard.h"
#include <zeno/utils/scope_exit.h>

class GraphsManagment;
class ZenoMainWindow;
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
class ZTcpServer;
#endif

class ZenoApplication : public QApplication
{
	Q_OBJECT
public:
    ZenoApplication(int &argc, char **argv);
    ~ZenoApplication();
    GraphsManagment *graphsManagment() const;
    void initFonts();
    void initStyleSheets();
    ZenoMainWindow* getMainWindow();
	QWidget* getWindow(const QString& objName);
    std::shared_ptr<ZCacheMgr> cacheMgr() const;
    std::shared_ptr<ProcessClipboard> procClipboard() const;
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    ZTcpServer* getServer();
#endif
    QStandardItemModel* logModel() const;

private:
    QString readQss(const QString& qssPath);
    void initMetaTypes();

    ZWidgetErrStream m_errSteam;
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    ZTcpServer* m_server;
#endif
    std::shared_ptr<ZCacheMgr> m_spCacheMgr;
    std::shared_ptr<ProcessClipboard> m_spProcClipboard;
    QDir m_appDataPath;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#define DlgInEventLoopScope                                                             \
    zeno::scope_exit sp([=]() { zenoApp->getMainWindow()->setInDlgEventLoop(false); }); \
    zenoApp->getMainWindow()->setInDlgEventLoop(true);

#endif
