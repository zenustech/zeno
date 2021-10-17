#pragma once


#include <QObject>
#include <QString>


class ApplicationData : public QObject {
    Q_OBJECT
public:
    explicit ApplicationData(QObject *parent = nullptr);
    Q_INVOKABLE void load_scene(QString str) const;
    Q_INVOKABLE QString get_descriptors() const;
    virtual ~ApplicationData();
};
