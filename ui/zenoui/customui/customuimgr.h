#ifndef __CUSTOMUI_MGR_H__
#define __CUSTOMUI_MGR_H__

#include <QtWidgets>

class CustomUIMgr
{
public:
    static CustomUIMgr& instance();
    void setNodeParamXml(const QString& nodeCls, const QString& xml);
    QString nodeParamXml(const QString& nodeCls);

private:
    CustomUIMgr();

    QMap<QString, QString> m_customParams;      //xml mapper.
};


#endif