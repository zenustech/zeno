#ifndef __ZENO_APPLICATION_H__
#define __ZENO_APPLICATION_H__

#include <QtWidgets>
#include "log/zwidgetostream.h"
#include "util/procclipboard.h"
#include <zeno/utils/scope_exit.h>

class GraphsManager;
class ZenoMainWindow;
class CalculationMgr;

class ZenoApplication : public QApplication
{
	Q_OBJECT
public:
    ZenoApplication(int &argc, char **argv);
    ~ZenoApplication();
    GraphsManager* graphsManager() const;
    CalculationMgr* calculationMgr() const;
    void initFonts();
    void initStyleSheets();
    ZenoMainWindow* getMainWindow();
	QWidget* getWindow(const QString& objName);
    std::shared_ptr<ProcessClipboard> procClipboard() const;
    QStandardItemModel* logModel() const;
    bool isUIApplication() const { return m_bUIApp; }

private slots:
    void onThreadLogReady(const QString& msg);

private:
    QString readQss(const QString& qssPath);
    void initMetaTypes();

    std::shared_ptr<ZWidgetErrStream> m_spUILogStream;
    std::shared_ptr<ProcessClipboard> m_spProcClipboard;
    CalculationMgr* m_calcMgr;
    QDir m_appDataPath;
    bool m_bUIApp;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#define DlgInEventLoopScope                                                             \
    zeno::scope_exit sp([=]() { zenoApp->getMainWindow()->setInDlgEventLoop(false); }); \
    zenoApp->getMainWindow()->setInDlgEventLoop(true);

#endif
