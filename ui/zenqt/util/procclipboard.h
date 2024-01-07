#ifndef __PROCESS_CLIPBOARD_H__
#define __PROCESS_CLIPBOARD_H__

#include <QString>

class ProcessClipboard
{
public:
    ProcessClipboard();
    void setCopiedAddress(const QString& addr);
    QString getCopiedAddress() const;

private:
    QString copiedAddress;
};

#endif