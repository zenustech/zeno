#ifndef __ZCODEEDITOR_H__
#define __ZCODEEDITOR_H__

#include <QCodeEditor>

class QSyntaxStyle;

class ZCodeEditor : public QCodeEditor
{
    Q_OBJECT
public:
    enum CodeHighLighter {
        HighLight_None,
        HighLight_CPP,
        HighLight_GLSL,
        HighLight_XML,
        HighLight_JSON,
        HighLight_LUA,
        HighLight_Python,
    };
    enum CodeCompleter {
        Completer_None,
        Completer_GLSL,
        Completer_LUA,
        Completer_Python,
    };
public:
    explicit ZCodeEditor(const QString& text, CodeHighLighter highLighter = HighLight_Python, CodeCompleter completer = Completer_Python, QWidget* parent = nullptr);
signals:
    void editFinished(const QString& text);
protected:
    void focusOutEvent(QFocusEvent* e) override;
private:
  void initUI();
  QSyntaxStyle* loadStyle(const QString& path);
  QCompleter* getCompleter(CodeCompleter completer);
  QStyleSyntaxHighlighter* getHighlighter(CodeHighLighter highLighter);
};

#endif
