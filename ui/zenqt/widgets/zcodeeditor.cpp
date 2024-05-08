#include "zcodeeditor.h"
#include <QGLSLCompleter>
#include <QLuaCompleter>
#include "ZPythonCompleter.h"
#include <QCXXHighlighter>
#include <QGLSLHighlighter>
#include <QXMLHighlighter>
#include <QJSONHighlighter>
#include <QLuaHighlighter>
#include <QPythonHighlighter>
#include <QSyntaxStyle>

ZCodeEditor::ZCodeEditor(const QString& text, CodeHighLighter highLighter, CodeCompleter completer, QWidget *parent)
    : QCodeEditor(parent)
{
    setCompleter(getCompleter(completer));
    setHighlighter(getHighlighter(highLighter));
    setText(text);
    initUI();
}

void ZCodeEditor::focusOutEvent(QFocusEvent* e)
{
    QCodeEditor::focusOutEvent(e);
    emit editFinished(toPlainText());
}

void ZCodeEditor::initUI()
{
    setSyntaxStyle(loadStyle(":/stylesheet/drakula.xml"));
    setWordWrapMode(QTextOption::WordWrap);
    setAutoIndentation(true);
    setTabReplace(true);
    setTabReplaceSize(4);
}

QCompleter* ZCodeEditor::getCompleter(CodeCompleter completer)
{
    switch (completer)
    {
    case ZCodeEditor::Completer_GLSL: return new QGLSLCompleter(this);
    case ZCodeEditor::Completer_LUA: return new QLuaCompleter(this);
    case ZCodeEditor::Completer_Python: return new ZPythonCompleter(this);
    default: return nullptr;
    }
}

QStyleSyntaxHighlighter* ZCodeEditor::getHighlighter(CodeHighLighter highLighter)
{
    switch (highLighter)
    {
    case HighLight_CPP: return new QCXXHighlighter;
    case HighLight_GLSL: return new QGLSLHighlighter ;
    case HighLight_XML: return new QXMLHighlighter ;
    case HighLight_JSON: return new QJSONHighlighter ;
    case HighLight_LUA: return new QLuaHighlighter ;
    case HighLight_Python: return new QPythonHighlighter ;
    }
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