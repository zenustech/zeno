#include "zfxtexttheme.h"

ZfxTextTheme::ZfxTextTheme()
{
	initDefaultTheme();
}

bool ZfxTextTheme::loadTheme(const QString& file)
{
	return false;
}

QTextCharFormat ZfxTextTheme::format(ZfxTextStyle category)
{
	return m_theme[category];
}

void ZfxTextTheme::initDefaultTheme()
{
	std::unordered_map<ZfxTextStyle, QString> colors = {
		{ZfxTextStyle::C_COMMENT, "#6a9955"},
		{ZfxTextStyle::C_FUNCTION, "#dcdcaa"},
		{ZfxTextStyle::C_LOCALVAR, "#9cdcfe"},
		{ZfxTextStyle::C_GLOBALVAR, "#4fc1ff"},
		{ZfxTextStyle::C_KEYWORD, "#569cd6"},
		{ZfxTextStyle::C_NUMBER, "#ae81ff"},
		{ZfxTextStyle::C_STRING, "#ce9178"}
	};

	for (const auto& [category, color] : colors) {
		QTextCharFormat format;
		format.setForeground(QBrush(QColor(color)));
		m_theme[category] = format;
	}
}
