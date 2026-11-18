#ifndef ZENO_APIMGR_H
#define ZENO_APIMGR_H

#include <QObject>
#include <QString>
#include <QJsonArray>
#include <atomic>
#include <thread>

class ZenoApplication;
class IGraphsModel;
class ZenoMainWindow;

class ApiMgr : public QObject
{
    Q_OBJECT

public:
    explicit ApiMgr(ZenoApplication* app, QObject* parent = nullptr);
    ~ApiMgr();

    void start();
    void stop();

    // cmdtool stdin/stdout protocol:
    // request:  {"type":"request","id":"...","name":"...","args":[...]}
    // response: {"type":"response","id":"...","ok":true,"result":"..."}
    // stdout is reserved for the response frames. Diagnostics use the normal
    // logging stream (stderr) so a client can reliably correlate a response.
    void sendResponse(const QString& requestId, bool ok, const QString& result);

    bool isRunning() const { return m_bRunning; }

signals:
    void commandReceived(const QString& requestId, const QString& name, const QJsonArray& args);

public slots:
    void handleCommand(const QString& requestId, const QString& name, const QJsonArray& args);

private slots:
    // Deferred runGraph bookkeeping: the reply to runGraph is sent only after
    // the runner process exits (ZenoMainWindow::runFinished).
    void onRunGraphFinished();
    void onRunnerError();
    void onStartProcFailed(const QString& reason);

private:
    void inputListenerThread();
    void handleFrame(const QString& line);

    void handleCreateGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleDeleteGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetNodeName(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleGetNodeSize(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleGraphGeometry(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleCreateNode(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleDeleteNode(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleAddLink(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleRemoveLink(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetParam(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleGetParam(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleListGraphs(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleListNodes(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleNodeInfo(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleGraphInfo(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleListNodeClasses(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleListNodeCategories(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleNodeClassesInfo(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleRunGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleKillGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetTimelineRange(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetView(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetOnce(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetNodePos(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);

    void handleSetNodePosList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetViewList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleAddLinkList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleDeleteNodeList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleCreateNodeList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetNodeNameList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleRemoveLinkList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetOnceList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetParamList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);

    void handleOpenFile(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSaveAs(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);

    void handleGetCameraOptix(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetCameraOptix(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleScreenShotOptix(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleGetOptixBackground(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleSetOptixBackground(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleGetOptixStatus(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);
    void handleOpenOptixViewport(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel);

    ZenoApplication* m_app;
    std::atomic<bool> m_bRunning;
    std::atomic<bool> m_running{true};
    std::thread* m_pThread;

    // Deferred runGraph response state (single run at a time; see
    // handleRunGraph).
    void finishPendingRun(bool ok, const QString& result);
    void connectRunSignalsOnce(ZenoMainWindow* main);
    // Run-scoped log capture: the runner's log lines are forwarded to
    // zenoApp->logModel() (the log panel's data source). beginRunLogCapture()
    // marks the first row of this run's output; collectRunErrorLogs() returns
    // the error-level rows appended since then. Level mapping (see
    // ZWidgetErrStream): "[E" -> QtFatalMsg, "[C" -> QtCriticalMsg (also the
    // fallback for unrecognized lines), "[W" -> QtWarningMsg; I/T/D rows are
    // not collected.
    void beginRunLogCapture();
    QString collectRunErrorLogs() const;
    // Deferred failure reply text: summary line plus the collected log rows.
    QString runFailureMessage(const QString& summary) const;
    QString m_pendingRunRequestId;
    int m_runLogStartRow = 0;
    bool m_runnerErrored = false;
    bool m_killRequested = false;
    bool m_runSignalsConnected = false;
};

#endif // ZENO_APIMGR_H
