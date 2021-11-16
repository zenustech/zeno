#ifndef __RENDER_PARAM_H__
#define __RENDER_PARAM_H__

#include <QtWidgets>

struct TextElement
{
	QFont font;
	QBrush fill;
	QString text;	//only used to template
	int x, y;
};

struct ImageElement
{
	QString id;
	QString type;
	QString image;
	QString imageOn;
	int x, y, w, h;
};

struct Component
{
	QVector<ImageElement> elements;
	TextElement text;
	QString id;
	int x, y, w, h;
};

struct HeaderParam
{
	Component name;
	Component status;
	Component control;
	Component display;
	Component backborad;
};

struct BodyParam
{
	Component leftTop;
	Component leftBottom;
	Component rightTop;
	Component rightBottom;

	Component backboard;
};

struct NodeParam
{
	HeaderParam header;
	BodyParam body;
};

#endif