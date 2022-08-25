#ifndef __ZTEXTEDIT_H__
#define __ZTEXTEDIT_H__

#include <QtWidgets>

class ZTextEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit ZTextEdit(QWidget* parent = nullptr);
    explicit ZTextEdit(const QString& text, QWidget* parent = nullptr);
    
signals:
    void editFinished();

protected:
    void focusOutEvent(QFocusEvent* e) override;
};


#endif