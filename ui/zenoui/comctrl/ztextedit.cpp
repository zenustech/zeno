#include "ztextedit.h"


ZTextEdit::ZTextEdit(QWidget* parent)
    : QTextEdit(parent)
{
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
}

ZTextEdit::ZTextEdit(const QString& text, QWidget* parent)
    : QTextEdit(text, parent)
{
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
}

QSize ZTextEdit::minimumSizeHint() const
{
    QSize minSz = QTextEdit::minimumSizeHint();
    return minSz;
}

QSize ZTextEdit::sizeHint() const
{
    QSize s(document()->size().toSize());
    /*
     * Make sure width and height have `usable' values.
     */
    s.rwidth() = std::max(256, s.width());
    s.rheight() = std::max(256, s.height());
    return s;
}

void ZTextEdit::focusOutEvent(QFocusEvent* e)
{
    QTextEdit::focusOutEvent(e);
    emit editFinished();
}

void ZTextEdit::resizeEvent(QResizeEvent* event)
{
    QSize s(document()->size().toSize());
    updateGeometry();
    QTextEdit::resizeEvent(event);
}