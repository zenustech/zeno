#ifndef __RENDER_PARAM_H__
#define __RENDER_PARAM_H__

#include <QtWidgets>

struct TextElement
{
    QString id;
	QFont font;
	QBrush fill;
	QString text;	//only used to template
    QRectF rc;
};

struct ImageElement
{
	QString id;
	QString type;
	QString image;
    QString imageHovered;
	QString imageOn;
    QRectF rc;
};

struct Component
{
    QString id;
	QVector<ImageElement> elements;
	TextElement text;
    QRect rc;
};

struct BackgroundComponent
{
    QString id;
    ImageElement image;
    QRect rc;
};

struct SocketComponent
{
    ImageElement image;
    TextElement text;
    QRect rc;
    QString id;
};

struct TextComponent
{
    TextElement text;
    QRect rc;
    QString id;
};

struct StatusComponent
{
    ImageElement mute;
    ImageElement view;
    ImageElement prep;
    QRect rc;
    QString id;
};

struct HeaderParam
{
    TextComponent name;
    StatusComponent status;
	Component control;
    BackgroundComponent display;
    BackgroundComponent backboard;
};

struct BodyParam
{
    SocketComponent leftTopSocket;
    SocketComponent leftBottomSocket;
    SocketComponent rightTopSocket;
    SocketComponent rightBottomSocket;

	BackgroundComponent backboard;
};

struct NodeParam
{
	HeaderParam header;
	BodyParam body;
};

#endif