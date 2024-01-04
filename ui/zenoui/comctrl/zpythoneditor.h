#ifndef __ZPYTHON_EDITOR_H__
#define __ZPYTHON_EDITOR_H__

#include <Qsci/qsciscintilla.h>
#include <QtWidgets>

class ZPythonEditor : public QsciScintilla
{
    Q_OBJECT
public:
    explicit ZPythonEditor(QWidget *parent = nullptr);
    explicit ZPythonEditor(const QString &text, QWidget *parent = nullptr);
signals:
    void editingFinished();
protected:
    bool eventFilter(QObject* watch, QEvent* event) override;
private:
  void initUI();
};

#endif
