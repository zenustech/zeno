#ifndef __ZENO_APPLICATION_H__
#define __ZENO_APPLICATION_H__

#include <QtWidgets>
#include "log/zwidgetostream.h"
#include "util/procclipboard.h"
#include "uicommon.h"
#include <zeno/utils/scope_exit.h>
#include <QQmlApplicationEngine>


class GraphsManager;
class ZenoMainWindow;
class CalculationMgr;
class NodeCateModel;
class MenuEventFilter;

class ZenoApplication : public QApplication
{
	Q_OBJECT
public:
    ZenoApplication(int &argc, char **argv);
    ~ZenoApplication();
    GraphsManager* graphsManager() const;
    CalculationMgr* calculationMgr() const;
    NodeCateModel* nodecateModel() const;
    void initFonts();
    void initStyleSheets();
    void initQuickQanavas();
    ZenoMainWindow* getMainWindow();
    QQmlApplicationEngine* getQmlEngine() const;
	QWidget* getWindow(const QString& objName);
    std::shared_ptr<ProcessClipboard> procClipboard() const;
    QStandardItemModel* logModel() const;
    bool isUIApplication() const { return m_bUIApp; }
    QString readQss(const QString& qssPath);
    void cleanQmlEngine();

private slots:
    void onThreadLogReady(const QString& msg);

private:
    void initMetaTypes();

    std::shared_ptr<ZWidgetErrStream> m_spUILogStream;
    std::shared_ptr<ProcessClipboard> m_spProcClipboard;
    GraphsManager* m_graphsMgr;
    NodeCateModel* m_nodecates;
    MenuEventFilter* m_menuEventFilter;
    CalculationMgr* m_calcMgr;
    QQmlApplicationEngine* m_engine;
    QDir m_appDataPath;
    bool m_bUIApp;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#define DlgInEventLoopScope                                                             \
    zeno::scope_exit sp([=]() { zenoApp->getMainWindow()->setInDlgEventLoop(false); }); \
    zenoApp->getMainWindow()->setInDlgEventLoop(true);

#endif
