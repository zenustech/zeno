#ifndef __ZCEFNODESEDITOR_H__
#define __ZCEFNODESEDITOR_H__

//#include <QtWidgets>
#include <QCefView.h>

class ZCefNodesEditor : public QCefView
{
    Q_OBJECT
public:
    ZCefNodesEditor(const QString url, const QCefSetting *setting, QWidget *parent = 0);
    ~ZCefNodesEditor();


};


#endif