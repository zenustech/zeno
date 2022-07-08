#ifndef __ZVALIDATOR_H__
#define __ZVALIDATOR_H__

#include <QtWidgets>

class PathValidator : public QValidator
{
    Q_OBJECT
public:
    explicit PathValidator(QObject *parent = nullptr);
    State validate(QString&, int&) const override;
    void fixup(QString&) const override;
};

class FilePathValidator : public QValidator
{
    Q_OBJECT
public:
    explicit FilePathValidator(QObject *parent = nullptr);
    State validate(QString&, int&) const override;
    void fixup(QString &) const override;
};


#endif