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
    QStandardItemModel* m_logModel;
    ZWidgetErrStream steam;
    bool m_bIOProcessing;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#endif