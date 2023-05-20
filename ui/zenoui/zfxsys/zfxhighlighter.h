#pragma once

#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QTextDocument>
#include <QRegularExpression>
#include "zfxtexttheme.h"


class ZfxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT

public:
    ZfxHighlighter(QTextDocument *parent = 0);

protected:
    void highlightBlock(const QString &text) override;

private:
    struct HighlightingRule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    QVector<HighlightingRule> m_highlightingRules;
    ZfxTextTheme m_highlightTheme;
};
