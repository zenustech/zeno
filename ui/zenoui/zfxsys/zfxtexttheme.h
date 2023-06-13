#pragma once

#include <unordered_map>
#include <QString>
#include <QTextCharFormat>

enum ZfxTextStyle
{
	C_NUMBER,
	C_STRING,
	C_LOCALVAR,
	C_GLOBALVAR,
	C_KEYWORD,
	C_FUNCTION,
	C_COMMENT
};

class ZfxTextTheme
{
public:
	ZfxTextTheme();
	bool loadTheme(const QString& file);
	QTextCharFormat format(ZfxTextStyle category);
private:
	void initDefaultTheme();

	std::unordered_map<ZfxTextStyle, QTextCharFormat> m_theme;
};
