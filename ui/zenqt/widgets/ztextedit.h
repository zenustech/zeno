#ifndef __ZTEXTEDIT_H__
#define __ZTEXTEDIT_H__

#include <QtWidgets>

class ZTextEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit ZTextEdit(QWidget* parent = nullptr);
    explicit ZTextEdit(const QString& text, QWidget* parent = nullptr);
    void setNodeIdx(const QModelIndex& index);
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

    QPersistentModelIndex m_index;
};


#endif