#include "zcodeeditor.h"
#include <QGLSLCompleter>
#include <QLuaCompleter>
#include "ZPythonCompleter.h"
#include <QZfxHighlighter>
#include <QGLSLHighlighter>
#include <QXMLHighlighter>
#include <QJSONHighlighter>
#include <QLuaHighlighter>
#include <QPythonHighlighter>
#include <QSyntaxStyle>
#include <zeno/core/data.h>
#include <zeno/core/Session.h>
#include <zeno/core/FunctionManager.h>

ZCodeEditor::ZCodeEditor(const QString& text, QWidget *parent)
    : QCodeEditor(parent), m_zfxHighLighter(new QZfxHighlighter), m_descLabel(nullptr)
{
    setCompleter(new ZPythonCompleter(this));
    setHighlighter(m_zfxHighLighter);
    setText(text);
    initUI();

    connect(this, &QTextEdit::cursorPositionChanged, this, &ZCodeEditor::slt_showFuncDesc);
}

void ZCodeEditor::setFuncDescLabel(ZenoFuncDescriptionLabel* descLabel)
{
    m_descLabel = descLabel;
}

void ZCodeEditor::focusOutEvent(QFocusEvent* e)
{
    QCodeEditor::focusOutEvent(e);
    Qt::FocusReason reason = e->reason();
    if (reason != Qt::ActiveWindowFocusReason)
        emit editFinished(toPlainText());
}

void ZCodeEditor::slt_showFuncDesc()
{
    if (!m_descLabel)
        return;

    QTextCursor cursor = textCursor();
    cursor.movePosition(QTextCursor::StartOfLine);
    cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
    QString currLine = cursor.selectedText();
    int positionInLine = textCursor().positionInBlock();

    QRegularExpression functionPattern = (QRegularExpression(R"(\b([_a-zA-Z][_a-zA-Z0-9]*\s+)?((?:[_a-zA-Z][_a-zA-Z0-9]*\s*::\s*)*[_a-zA-Z][_a-zA-Z0-9]*)(?=\s*\())"));
    auto matchIterator = functionPattern.globalMatch(currLine);
    while (matchIterator.hasNext())
    {
        auto match = matchIterator.next();
        if (match.capturedStart(2) + match.capturedLength(2) + 1 == positionInLine) {
            zeno::FUNC_INFO info = zeno::getSession().funcManager->getFuncInfo(currLine.mid(match.capturedStart(2), match.capturedLength(2)).toStdString());
            if (!info.name.empty()) {
                m_descLabel->setDesc(info, 0);
                m_descLabel->setCurrentFuncName(info.name);

                QFontMetrics metrics(this->font());
                const QPoint& parentGlobalPos = m_descLabel->getPropPanelPos();
                QPoint pos = mapToGlobal(QPoint(0, 0));
                pos.setX(pos.x() - parentGlobalPos.x() + metrics.width(currLine.left(positionInLine)) + 20);
                pos.setY(pos.y() - parentGlobalPos.y() + (cursor.blockNumber() + 1) * metrics.height() + 5);
                m_descLabel->move(pos);
                m_descLabel->show();
                return;
            }
        }
    }
    if (m_descLabel->isVisible()) {
        m_descLabel->hide();
    }
}

void ZCodeEditor::initUI()
{
    setSyntaxStyle(loadStyle(":/stylesheet/drakula.xml"));
    setWordWrapMode(QTextOption::WordWrap);
    setAutoIndentation(true);
    setTabReplace(true);
    setTabReplaceSize(4);
}

QSyntaxStyle* ZCodeEditor::loadStyle(const QString& path)
{
    QFile fl(path);

    if (!fl.open(QIODevice::ReadOnly))
    {
        return QSyntaxStyle::defaultStyle();
    }

    auto style = new QSyntaxStyle(this);

    if (!style->load(fl.readAll()))
    {
        delete style;
        return QSyntaxStyle::defaultStyle();
    }

    return style;
}