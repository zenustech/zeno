#include "ztextedit.h"


ZTextEdit::ZTextEdit(QWidget* parent)
    : QTextEdit(parent)
{
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

ZTextEdit::ZTextEdit(const QString& text, QWidget* parent)
    : QTextEdit(text, parent)
{
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
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
    s.rwidth() = std::max(100, s.width());
    s.rheight() = std::max(100, s.height());
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