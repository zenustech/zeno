#include "ztextedit.h"


ZTextEdit::ZTextEdit(QWidget* parent)
    : QTextEdit(parent), m_realLineCount(0)
{
    initUI();
}

ZTextEdit::ZTextEdit(const QString& text, QWidget* parent)
    : QTextEdit(text, parent), m_realLineCount(0)
{
    initUI();
}

void ZTextEdit::setNodeIdx(const QModelIndex& index) {
    m_index = index;
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
    connect(pTextDoc, &QTextDocument::contentsChanged, this, [this]() {
        QFontMetrics metrics(font());
        int currline = qCeil(static_cast<double>(document()->size().height()) / metrics.lineSpacing());
        if (currline != m_realLineCount) {
            emit lineCountReallyChanged(m_realLineCount, currline);
            m_realLineCount = currline;
        }
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