#ifndef __ZPTHON_EDITOR_ITEM_H__
#define __ZPTHON_EDITOR_ITEM_H__

#include <QtWidgets>
#include "zenoparamwidget.h"

class QsciScintilla;

class ZPythonEditorItem : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZPythonEditorItem(QGraphicsItem* parent = nullptr);
    ZPythonEditorItem(const QString& value, LineEditParam param, QGraphicsItem* parent = nullptr);
    QString text() const;
    void setText(const QString& text);

protected:
    bool eventFilter(QObject* object, QEvent* event) override;

signals:
    void textChanged();
    void editingFinished();

private:
    QString m_value;
    QsciScintilla* m_pTextEdit;
};

#endif