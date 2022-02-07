#ifndef __ZENO_APPLICATION_H__
#define __ZENO_APPLICATION_H__

#include <QtWidgets>

class GraphsManagment;

class GraphsModel;

class ZenoApplication : public QApplication
{
	Q_OBJECT
public:
    ZenoApplication(int &argc, char **argv);
    ~ZenoApplication();
    QSharedPointer<GraphsManagment> graphsManagment() const;
    void initFonts();

private:
    QSharedPointer<GraphsManagment> m_pGraphs;
};

#define zenoApp (qobject_cast<ZenoApplication*>(QApplication::instance()))

#endif