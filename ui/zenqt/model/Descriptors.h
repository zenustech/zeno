#ifndef __DESCRIPTORS_H__
#define __DESCRIPTORS_H__

#include "../common.h"

class Descriptors : public QObject
{
    Q_OBJECT
public:
    static Descriptors* instance();

    NODE_DESCRIPTOR getDescriptor(const QString& name) const;

private:
    Descriptors();
    void initDescs();

    QMap<QString, NODE_DESCRIPTOR> m_descs;
};

#endif