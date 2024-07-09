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
    void setFuncDescLabel(ZenoFuncDescriptionLabel* descLabel);
signals:
    void editFinished(const QString& text);
protected:
    void focusOutEvent(QFocusEvent* e) override;

private slots:
    void slt_showFuncDesc();
private:
  void initUI();
  QSyntaxStyle* loadStyle(const QString& path);

  QZfxHighlighter* m_zfxHighLighter;
  ZenoFuncDescriptionLabel* m_descLabel;
};

#endif
