#pragma once

#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QRegularExpression>
#include "zfxtexttheme.h"

class QTextEdit;

class ZfxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT

public:
    ZfxHighlighter(QTextEdit *textEdit = nullptr);

protected:
    void highlightBlock(const QString &text) override;
    bool eventFilter(QObject* object, QEvent* event) override;

private:
    void initRules();
    void highlightCurrentLine(QList<QTextEdit::ExtraSelection>& extraSelections);
    void highlightParenthesis(QList<QTextEdit::ExtraSelection>& extraSelections);
    void onCursorPositionChanged();
    void onSelectionChanged();

    struct HighlightingRule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    QVector<HighlightingRule> m_highlightingRules;
    ZfxTextTheme m_highlightTheme;
    QTextEdit* m_pTextEdit;
};
