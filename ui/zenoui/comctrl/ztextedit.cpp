#include "ztextedit.h"

static QVector<QPair<QString, QString>> parentheses = {
    {"(", ")"},
    {"{", "}"},
    {"[", "]"},
    {"\"", "\""},
    {"'", "'"}
};

ZTextEdit::ZTextEdit(QWidget* parent)
    : QTextEdit(parent)
    , m_autoParentheses(true)
{
    initUI();
}

ZTextEdit::ZTextEdit(const QString& text, QWidget* parent)
    : QTextEdit(text, parent)
    , m_autoParentheses(true)
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

void ZTextEdit::setAutoParentheses(bool enabled)
{
    m_autoParentheses = enabled;
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

void ZTextEdit::resizeEvent(QResizeEvent* e)
{
    QSize s(document()->size().toSize());
    QTextEdit::resizeEvent(e);
    updateGeometry();
    emit geometryUpdated();
}

void ZTextEdit::keyPressEvent(QKeyEvent* e)
{
    QTextEdit::keyPressEvent(e);

    if (m_autoParentheses) {
        for (const auto& [left, right] : parentheses) {
            // if ( : auto add right
            if (left == e->text()) {
                insertPlainText(right);
                moveCursor(QTextCursor::MoveOperation::Left);
                break;
            }
            // if ()) : auto delete right
            else if (right == e->text()) {
                auto symbol = textUnderCursor();
                if (symbol.right(1) == right) {
                    textCursor().deletePreviousChar();
                    moveCursor(QTextCursor::MoveOperation::Right);
                }
                break;
            }
        }
    }
}

QString ZTextEdit::textUnderCursor()
{
    QTextCursor tc = textCursor();
    tc.select(QTextCursor::WordUnderCursor);
    return tc.selectedText();
}
