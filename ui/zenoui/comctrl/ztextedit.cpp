#include "ztextedit.h"


ZTextEdit::ZTextEdit(QWidget* parent)
    : QTextEdit(parent)
{
    initUI();
}

ZTextEdit::ZTextEdit(const QString& text, QWidget* parent)
    : QTextEdit(text, parent)
{
    initUI();
}

void ZTextEdit::initUI()
{
    setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    QTextDocument *pTextDoc = document();
    connect(pTextDoc, &QTextDocument::contentsChanged, this, [=]() {
        QSize s(document()->size().toSize());
        updateGeometry();
        emit geometryUpdated();
    });
}

QSize ZTextEdit::minimumSizeHint() const
{
    QSize minSz = QTextEdit::minimumSizeHint();
    return minSz;
}

QSize ZTextEdit::sizeHint() const
{
    QSize sz = QTextEdit::sizeHint();
    return sz;
}

QSize ZTextEdit::viewportSizeHint() const
{
    QSize sz = document()->size().toSize();
    return sz;
}

void ZTextEdit::focusInEvent(QFocusEvent* e)
{
    QTextEdit::focusInEvent(e);
}

void ZTextEdit::focusOutEvent(QFocusEvent* e)
{
    QTextEdit::focusOutEvent(e);
    emit editFinished();
}

void ZTextEdit::resizeEvent(QResizeEvent* event)
{
    QSize s(document()->size().toSize());
    QTextEdit::resizeEvent(event);
    updateGeometry();
    emit geometryUpdated();
}