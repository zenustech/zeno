#include "zpythoneditoritem.h"
#include <Qsci/qsciscintilla.h>
#include <Qsci/qscilexerpython.h>
#include <Qsci/qsciapis.h>
#include <zenoui/style/zenostyle.h>


ZPythonEditorItem::ZPythonEditorItem(QGraphicsItem* parent)
    : ZenoParamWidget(parent)
    , m_pTextEdit(nullptr)
{
    m_pTextEdit = new QsciScintilla;
    QsciLexerPython* textLexer = new QsciLexerPython;
    m_pTextEdit->setLexer(textLexer);
    m_pTextEdit->setMarginLineNumbers(10, true);

    m_pTextEdit->setTabWidth(4);
    m_pTextEdit->setMarginType(0, QsciScintilla::NumberMargin);
    m_pTextEdit->setMarginWidth(0, 20);
    m_pTextEdit->setMinimumWidth(ZenoStyle::dpiScaled(300));

    setWidget(m_pTextEdit);
}

ZPythonEditorItem::ZPythonEditorItem(const QString& value, LineEditParam param, QGraphicsItem* parent)
{
    m_pTextEdit = new QsciScintilla;
    QsciLexerPython* textLexer = new QsciLexerPython;
    m_pTextEdit->setLexer(textLexer);
    m_pTextEdit->setMarginLineNumbers(10, true);
    m_pTextEdit->setText(value);

    m_pTextEdit->setTabWidth(4);
    m_pTextEdit->setMarginType(0, QsciScintilla::NumberMargin);
    m_pTextEdit->setMarginWidth(0, 20);
    m_pTextEdit->setMinimumWidth(ZenoStyle::dpiScaled(300));

    //QsciAPIs* apis = new QsciAPIs(textLexer);
    //apis->add(QString("import"));
    //apis->prepare();

    //m_pTextEdit->setAutoCompletionSource(QsciScintilla::AcsAll);
    //m_pTextEdit->setAutoCompletionCaseSensitivity(true);
    //m_pTextEdit->setAutoCompletionThreshold(1);

    setWidget(m_pTextEdit);
}

QString ZPythonEditorItem::text() const
{
    return m_pTextEdit->text();
}

void ZPythonEditorItem::setText(const QString& text)
{
    m_pTextEdit->setText(text);
}

bool ZPythonEditorItem::eventFilter(QObject* object, QEvent* event)
{
    if (object == m_pTextEdit && event->type() == QEvent::FocusOut)
    {
        emit editingFinished();
    }
    return ZenoParamWidget::eventFilter(object, event);
}