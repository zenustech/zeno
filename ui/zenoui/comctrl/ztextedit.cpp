#include "ztextedit.h"


ZTextEdit::ZTextEdit(QWidget* parent)
    : QTextEdit(parent)
{
}

ZTextEdit::ZTextEdit(const QString& text, QWidget* parent)
    : QTextEdit(text, parent)
{
}

void ZTextEdit::focusOutEvent(QFocusEvent* e)
{
    QTextEdit::focusOutEvent(e);
    emit editFinished();
}