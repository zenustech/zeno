#ifndef __RENDER_PARAM_H__
#define __RENDER_PARAM_H__

#include <QtWidgets>

struct TextElement
{
    QString id;
	QFont font;
	QBrush fill;
    QBrush background;
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

struct ImageComponent
{
    QString id;
    ImageElement image;
    QRect rc;
};

struct BackgroundComponent
{
    QString id;
    ImageElement imageElem;
    QRect rc;

    QColor clr_normal, clr_hovered, clr_selected;
    int lt_radius, rt_radius, lb_radius, rb_radius;
    bool bApplyImage;
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
    ImageComponent display;
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

struct LineEditParam
{
    QFont font;
    QPalette palette;
    QMargins margins;
};

struct ComboBoxParam
{
    QFont font;
    QColor itemBgNormal;
    QColor itemBgHovered;
    QColor itemBgSelected;
    QColor textColor;
    QPalette palette;   //for lineedit part.
};

struct NodeUtilParam
{
    //header
    //QRectF rcHeaderBg;
    //set left corner of header background as origin, always (0,0).
    BackgroundComponent headerBg;

    QRectF rcMute, rcView, rcPrep;
    ImageElement mute, view, prep;

    QRectF rcCollasped;
    ImageElement collaspe;

    QPointF namePos;
    TextElement name;

    //body
    //QRectF rcBodyBg;
    BackgroundComponent bodyBg;

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
    LineEditParam lineEditParam;
    ComboBoxParam comboboxParam;
};

#endif