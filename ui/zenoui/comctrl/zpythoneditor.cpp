#include "zpythoneditor.h"
#include <Qsci/qscilexerpython.h>
#include <Qsci/qsciapis.h>
#include <zenoui/style/zenostyle.h>

ZPythonEditor::ZPythonEditor(QWidget *parent)
    : QsciScintilla(parent)
{
    initUI();
}

ZPythonEditor::ZPythonEditor(const QString &text, QWidget *parent)
    : QsciScintilla(parent)
{
    initUI();
    setText(text);
}

bool ZPythonEditor::eventFilter(QObject* watch, QEvent* event)
{
    if (watch == this && event->type() == QEvent::FocusOut)
    {
        emit editingFinished();
    }
    return QsciScintilla::eventFilter(watch, event);
}

void ZPythonEditor::initUI()
{
    QsciLexerPython* textLexer = new QsciLexerPython(this);
    setLexer(textLexer);
    setMarginLineNumbers(10, true);

    setTabWidth(4);
    setMarginType(0, QsciScintilla::NumberMargin);
    setMarginWidth(0, 30);
    setMinimumWidth(ZenoStyle::dpiScaled(300));
    setAutoCompletionSource(QsciScintilla::AcsAll);
    setAutoCompletionCaseSensitivity(true);
    setAutoCompletionThreshold(1);

    QsciAPIs* apis = new QsciAPIs(textLexer);
    QStringList lst = { "import", "graph", "createGraph", "removeGraph", "forkGraph", "forkMaterial", "renameGraph"
    , "createNode", "deleteNode", "node", "addLink",  "removeLink", "name"
    ,"objCls", "ident"};
    for(const auto& key : lst)
        apis->add(key);
    apis->prepare();
    textLexer->setAPIs(apis);
    installEventFilter(this);
}
