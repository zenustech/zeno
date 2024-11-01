#ifndef __ZTEXTEDIT_H__
#define __ZTEXTEDIT_H__

#include <QtWidgets>

class ZenoHintListWidget;
class ZenoFuncDescriptionLabel;

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

    void setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabl);
    void hintSelectedSetText(QString itemSelected);

public slots:
    void sltHintSelected(QString itemSelected);
    void sltSetFocus();

signals:
    void editFinished();
    void geometryUpdated();
    void lineCountReallyChanged(int, int);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void focusInEvent(QFocusEvent* e) override;
    void focusOutEvent(QFocusEvent* e) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void initUI();

    QPersistentModelIndex m_index;

    int m_realLineCount;

    bool m_bShowHintList;
    QString m_firstCandidateWord;
    ZenoHintListWidget* m_hintlist;
    ZenoFuncDescriptionLabel* m_descLabel;
};


#endif