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
	std::unordered_map<ZfxTextStyle, std::pair<QString, QString>> colors = {
		{ZfxTextStyle::C_COMMENT, {"#6a9955", ""}},
		{ZfxTextStyle::C_FUNCTION, {"#dcdcaa", ""}},
		{ZfxTextStyle::C_LOCALVAR, {"#9cdcfe", ""}},
		{ZfxTextStyle::C_GLOBALVAR, {"#4fc1ff", ""}},
		{ZfxTextStyle::C_KEYWORD, {"#569cd6", ""}},
		{ZfxTextStyle::C_NUMBER, {"#ae81ff", ""}},
		{ZfxTextStyle::C_STRING, {"#ce9178", ""}},
		{ZfxTextStyle::C_CURRENTLINE, {"", "#383b4c"}},
		{ZfxTextStyle::C_PARENTHESES, {"#f1a119", "#264f78"}}
	};

	for (const auto& [category, color] : colors) {
		auto [fcolor, bcolor] = color;
		QTextCharFormat format;
		if (!fcolor.isEmpty()) {
			format.setForeground(QBrush(QColor(fcolor)));
		}
		if (!bcolor.isEmpty()) {
			format.setBackground(QBrush(QColor(bcolor)));
		}
		m_theme[category] = format;
	}
}
