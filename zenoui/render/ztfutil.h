#ifndef __ZTF_UTIL_H__
#define __ZTF_UTIL_H__

#include "renderparam.h"
#include <rapidxml/rapidxml_print.hpp>

using namespace rapidxml;

typedef rapidxml::xml_node<>* XML_NODE;
typedef rapidxml::xml_document<>& XMLDOC_REF;

class ZtfUtil
{
public:
	static ZtfUtil& GetInstance();
	NodeParam loadZtf(const QString& filename);
    NodeUtilParam toUtilParam(const NodeParam& nodeParam);
    void exportZtf(NodeParam param /*, LineParam param2, BackgroundParam param3*/, const QString& output);

private:
	ZtfUtil();
	Component _parseComponent(rapidxml::xml_node<>*);
	TextElement _parseText(rapidxml::xml_node<>*);
	ImageElement _parseImage(rapidxml::xml_node<>*);

	const char *qsToString(const QString &qs);

	XML_NODE _exportHeader(rapidxml::xml_document<> &doc, HeaderParam headerParam);
    XML_NODE _exportBody(rapidxml::xml_document<> &doc, BodyParam headerParam);
    XML_NODE _exportText(XMLDOC_REF doc, TextElement textElem);
    XML_NODE _exportImage(XMLDOC_REF doc, ImageElement imageElem);

	void _exportRc(rapidxml::xml_document<>& doc, XML_NODE node, QRect rc);
    void _exportRcF(rapidxml::xml_document<>& doc, XML_NODE node, QRectF rc);
};

#endif