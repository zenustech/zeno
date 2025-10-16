#ifndef __ZCODEEDITOR_H__
#define __ZCODEEDITOR_H__

#include <QCodeEditor>
#include "panel/ZenoHintListWidget.h"

class QSyntaxStyle;
class QZfxHighlighter;

class ZCodeEditor : public QCodeEditor
{
    Q_OBJECT
public:
    explicit ZCodeEditor(const QString& text, QWidget* parent = nullptr);
    void setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabel);
    void setNodeIndex(QModelIndex nodeIdx);

signals:
    void editFinished(const QString& text);

protected:
    void focusInEvent(QFocusEvent* e) override;
    void focusOutEvent(QFocusEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private slots:
    void slt_showFuncDesc();
    void sltHintSelected(QString itemSelected);
    void sltSetFocus();

private:
  void initUI();
  QSyntaxStyle* loadStyle(const QString& path);
  void hintSelectedSetText(QString text);

  QZfxHighlighter* m_zfxHighLighter;
  ZenoHintListWidget* m_hintlist;
  ZenoFuncDescriptionLabel* m_descLabel;
  QString m_firstCandidateWord;

  QModelIndex m_nodeIdx;

  int minimumHeight = 220;
};

#endif
