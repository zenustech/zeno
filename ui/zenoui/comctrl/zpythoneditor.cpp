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
    QStringList lst  = { "import", "zeno.graph(str subg_name)", "zeno.createGraph(str subg_name, int type = 0)", "zeno.removeGraph(str subg_name)", "zeno.forkGraph(str subg_name, str node_ident)", "zeno.forkMaterial(str preset_subg_name, str subg_name, str mtlid)", "zeno.renameGraph(str subg_name, str new_name)"
    , "createNode(str node_class, *kwargs)", "deleteNode(str node_ident)", "node(str node_ident)", "addLink(str out_node_ident, str out_node_socket, str in_node_ident, str in_node_socket)",  "removeLink(str out_node_ident, str out_node_socket, str in_node_ident, str in_node_socket)", "name()"
    ,"objCls", "ident", "view", "mute", "once", "fold"};
    for(const auto& key : lst)
        apis->add(key);
    apis->prepare();
    textLexer->setAPIs(apis);
    installEventFilter(this);
}
