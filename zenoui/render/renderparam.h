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

struct DistanceParam
{
    int paramsVPadding = 10;        //the first param to background's top.
    int paramsVSpacing = 16;
    int paramsBottomPadding = 10;   //the dist from last param to next item(socket).
    int paramsLPadding = 16;
    int paramsToTopSocket = 16;
};

struct NodeUtilParam
{
    //header
    QRectF rcHeaderBg;//set left corner as origin, always (0,0).
    ImageElement headerBg;

    QRectF rcMute, rcView, rcPrep;
    ImageElement mute, view, prep;

    QRectF rcCollasped;
    ImageElement collaspe;

    QPointF namePos;
    TextElement name;

    //body
    QRectF rcBodyBg;
    ImageElement bodyBg;

    QSizeF szSocket;
    ImageElement socket;

    qreal socketHOffset;
    qreal socketToText;
    qreal socketVMargin;

    QBrush nameClr;
    QFont nameFont;

    QBrush socketClr;
    QFont socketFont;

    QBrush paramClr;
    QFont paramFont;

    DistanceParam distParam;
};

#endif