#ifndef __ZTEXTEDIT_H__
#define __ZTEXTEDIT_H__

#include <QtWidgets>

class ZTextEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit ZTextEdit(QWidget* parent = nullptr);
    explicit ZTextEdit(const QString& text, QWidget* parent = nullptr);
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;
    QSize viewportSizeHint() const override;

signals:
    void editFinished();
    void geometryUpdated();

protected:
    void focusInEvent(QFocusEvent* e) override;
    void focusOutEvent(QFocusEvent* e) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void initUI();
};


#endif