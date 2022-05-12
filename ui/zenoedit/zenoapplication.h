#ifndef __ZENO_APPLICATION_H__
#define __ZENO_APPLICATION_H__

#include <QtWidgets>
#include "zwidgetostream.h"

class GraphsManagment;
class ZenoMainWindow;

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
    QStandardItemModel* logModel() const;

private:
    QString readQss(const QString& qssPath);

    QSharedPointer<GraphsManagment> m_pGraphs;
    ZWidgetErrStream m_errSteam;
    bool m_bIOProcessing;
    QDir m_appDataPath;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#endif
