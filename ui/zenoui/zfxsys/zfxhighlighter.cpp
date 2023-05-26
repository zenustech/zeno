#include <QTextEdit>
#include "zfxhighlighter.h"
#include "zfxkeywords.h"

ZfxHighlighter::ZfxHighlighter(QTextEdit* textEdit)
	: QSyntaxHighlighter(textEdit)
	, m_pTextEdit(textEdit)
{
	initRules();
	if (!m_pTextEdit) return;
	m_pTextEdit->installEventFilter(this);
	connect(m_pTextEdit, &QTextEdit::cursorPositionChanged, this, &ZfxHighlighter::onCursorPositionChanged);
	connect(m_pTextEdit, &QTextEdit::selectionChanged, this, &ZfxHighlighter::onSelectionChanged);
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

void ZfxHighlighter::initRules()
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
	rule.pattern = QRegularExpression(QString("%1[^\n]*").arg(zfxComment));
	rule.format = m_highlightTheme.format(ZfxTextStyle::C_COMMENT);
	m_highlightingRules.append(rule);
}

void ZfxHighlighter::highlightCurrentLine(QList<QTextEdit::ExtraSelection>& extraSelections)
{
	if (m_pTextEdit->isReadOnly()) return;
	
	QTextEdit::ExtraSelection selection;
	selection.format = m_highlightTheme.format(ZfxTextStyle::C_CURRENTLINE);
	selection.format.setProperty(QTextFormat::FullWidthSelection, true);
	selection.cursor = m_pTextEdit->textCursor();
	selection.cursor.clearSelection();
	extraSelections.append(selection);
	
}

void ZfxHighlighter::highlightParenthesis(QList<QTextEdit::ExtraSelection>& extraSelections)
{
	auto cursor = m_pTextEdit->textCursor();
	auto curChar = m_pTextEdit->document()->characterAt(cursor.position());
	if (curChar.isNull()) return;

	for (const auto& [left, right] : zfxParentheses) {
		// find dir
		int dir = 0;
		QChar pairChar = curChar;
		if (curChar == left) {
			dir = 1;
			pairChar = right[0];
		}
		else if (curChar == right) {
			dir = -1;
			pairChar = left[0];
		}
		else {
			continue;
		}
		// try find the pair char position
		int curCharCnt = 1;
		int totalCharCnt = m_pTextEdit->document()->characterCount();
		int pos = cursor.position() + dir;
		for (; pos >= 0 && pos < totalCharCnt; pos += dir) {
			auto character = m_pTextEdit->document()->characterAt(pos);
			if (character == curChar) {
				curCharCnt++;
			}
			else if (character == pairChar) {
				curCharCnt--;
			}
			else if (character == zfxComment) {
				break;
			}
			if (curCharCnt == 0) break;
		}
		if (curCharCnt == 0) {
			QTextEdit::ExtraSelection selection;
			selection.format = m_highlightTheme.format(ZfxTextStyle::C_PARENTHESES);
			auto dirEnum = dir < 0 ? QTextCursor::Left : QTextCursor::Right;
			// the pair
			selection.cursor = cursor;
			selection.cursor.clearSelection();
			selection.cursor.movePosition(dirEnum, QTextCursor::MoveAnchor, std::abs(cursor.position() - pos));
			selection.cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor, 1);
			extraSelections.append(selection);
			// the cur
			selection.cursor = cursor;
			selection.cursor.clearSelection();
			selection.cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor, 1);
			extraSelections.append(selection);
		}
		break;
	}
}

void ZfxHighlighter::onCursorPositionChanged()
{
	if (!m_pTextEdit || !m_pTextEdit->hasFocus()) return;

	QList<QTextEdit::ExtraSelection> extraSelections;
	highlightCurrentLine(extraSelections);
	highlightParenthesis(extraSelections);
	m_pTextEdit->setExtraSelections(extraSelections);
}

void ZfxHighlighter::onSelectionChanged()
{
}

bool ZfxHighlighter::eventFilter(QObject* object, QEvent* event)
{
	if (object == m_pTextEdit) {
		if (event->type() == QEvent::FocusOut) {
			// clear selection
			QTextCursor cursor = m_pTextEdit->textCursor();
			cursor.clearSelection();
			m_pTextEdit->setTextCursor(cursor);
			m_pTextEdit->setExtraSelections(QList<QTextEdit::ExtraSelection>());
		}
		else if (event->type() == QEvent::FocusIn) {
			onCursorPositionChanged();
		}
	}
	return QSyntaxHighlighter::eventFilter(object, event);
}
