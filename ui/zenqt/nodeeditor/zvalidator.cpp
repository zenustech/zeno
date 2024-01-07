#include "zvalidator.h"


PathValidator::PathValidator(QObject* parent) : QValidator(parent)
{

}

QValidator::State PathValidator::validate(QString& input, int& pos) const
{
    if (input.isEmpty())
        return Acceptable;
    bool bValid = QFileInfo::exists(input);
    if (bValid)
        return Acceptable;
    return Intermediate;
}

void PathValidator::fixup(QString& wtf) const
{
    wtf = "";
}


FilePathValidator::FilePathValidator(QObject *parent)
    : QValidator(parent)
{
}

QValidator::State FilePathValidator::validate(QString& input, int& pos) const
{
    if (input.isEmpty())
        return Acceptable;
    QFileInfo info(input);
    if (info.isFile())
        return Acceptable;
    return Intermediate;
}

void FilePathValidator::fixup(QString& wtf) const
{
    wtf = "";
}