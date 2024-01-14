#include "zfxhighlighter.h"
#include "zfxkeywords.h"

ZfxHighlighter::ZfxHighlighter(QTextDocument* parent)
	: QSyntaxHighlighter(parent)
{
	HighlightingRule rule;

	// number
	rule.pattern = QRegularExpression("\\b\\d+(\\.\\d+)?\\b");
	rule.format = m_highlightTheme.format(ZfxTextStyle::C_NUMBER);
	m_highlightingRules.append(rule);

	// string, seems no need
	
	// local var
	rule.pattern = QRegularExpression("\\@[a-zA-Z]+\\b");
	rule.format = m_highlightTheme.format(ZfxTextStyle::C_LOCALVAR);
	m_highlightingRules.append(rule);

	// global var
	rule.pattern = QRegularExpression("\\$[a-zA-Z]+\\b");
	rule.format = m_highlightTheme.format(ZfxTextStyle::C_GLOBALVAR);
	m_highlightingRules.append(rule);
	
	// keyword
	
	// func, only highlight supported
	QStringList functionPatterns;
	for (const auto& func : zfxFunction) {
		functionPatterns << QString("\\b%1\\b").arg(func);
	}
	auto functionFormat = m_highlightTheme.format(ZfxTextStyle::C_FUNCTION);
	for (const auto& pattern : functionPatterns) {
		rule.pattern = QRegularExpression(pattern);
		rule.format = functionFormat;
		m_highlightingRules.append(rule);
	}

	// comment
	rule.pattern = QRegularExpression("#[^\n]*");
	rule.format = m_highlightTheme.format(ZfxTextStyle::C_COMMENT);
	m_highlightingRules.append(rule);
}

void ZfxHighlighter::highlightBlock(const QString& text)
{
	if (text.size() == 0) return;
    for (const HighlightingRule& rule : m_highlightingRules) {
        QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
        while (matchIterator.hasNext()) {
            QRegularExpressionMatch match = matchIterator.next();
            setFormat(match.capturedStart(), match.capturedLength(), rule.format);
        }
    }
}
