#include "procclipboard.h"


ProcessClipboard::ProcessClipboard()
{

}

void ProcessClipboard::setCopiedAddress(const QString& addr)
{
    copiedAddress = addr;
}

QString ProcessClipboard::getCopiedAddress() const
{
    return copiedAddress;
}