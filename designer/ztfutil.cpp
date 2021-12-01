#include "ztfutil.h"
#include "common_id.h"
#include <stdexcept>

ZtfUtil& ZtfUtil::GetInstance()
{
	static ZtfUtil instance;
	return instance;
}

ZtfUtil::ZtfUtil()
{
}

NodeParam ZtfUtil::loadZtf(const QString& filename)
{
	QFile file(filename);
	bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        throw std::runtime_error("ztf file: [" + filename.toStdString() + "] not found");
    }
	QByteArray arr = file.readAll();

	rapidxml::xml_document<> doc;
	doc.parse<0>(arr.data());
	auto root = doc.first_node();

	NodeParam param;
	for (auto child = root->first_node(); child != nullptr; child = child->next_sibling())
	{
		QString name(child->name());
		if (name == "header")
		{
			HeaderParam header;
			for (auto component = child->first_node(); component != nullptr; component = component->next_sibling())
			{
				Component comp = _parseComponent(component);
                if (comp.id == COMPONENT_STATUS)
				{
                    header.status.id = comp.id;
					header.status.mute = comp.elements[0];
                    header.status.view = comp.elements[1];
                    header.status.prep = comp.elements[2];
                    header.status.rc = comp.rc;
				} else if (comp.id == COMPONENT_CONTROL)
				{
					header.control = comp;
                } else if (comp.id == COMPONENT_DISPLAY) {
                    header.display.id = comp.id;
					header.display.image = comp.elements[0];
                    header.display.rc = comp.rc;
                } else if (comp.id == COMPONENT_HEADER_BG) {
                    header.backboard.id = comp.id;
                    header.backboard.image = comp.elements[0];
                    header.backboard.rc = comp.rc;
                } else if (comp.id == COMPONENT_NAME) {
                    header.name.id = comp.id;
                    header.name.text = comp.text;
                    header.name.rc = comp.rc;
				}
			}
			param.header = header;
		}
		else if (name == "body")
		{
			BodyParam body;
			for (auto component = child->first_node(); component != nullptr; component = component->next_sibling())
			{
				Component comp = _parseComponent(component);
                if (comp.id == COMPONENT_BODY_BG)
				{
                    body.backboard.image = comp.elements[0];
                    body.backboard.rc = comp.rc;
                    body.backboard.id = comp.id;
                } else if (comp.id == COMPONENT_LTSOCKET) {
                    body.leftTopSocket.image = comp.elements[0];
                    body.leftTopSocket.text = comp.text;
                    body.leftTopSocket.rc = comp.rc;
                    body.leftTopSocket.id = comp.id;
                } else if (comp.id == COMPONENT_LBSOCKET) {
					body.leftBottomSocket.image = comp.elements[0];
                    body.leftBottomSocket.text = comp.text;
                    body.leftBottomSocket.rc = comp.rc;
                    body.leftBottomSocket.id = comp.id;
                } else if (comp.id == COMPONENT_RTSOCKET) {
                    body.rightTopSocket.image = comp.elements[0];
                    body.rightTopSocket.text = comp.text;
                    body.rightTopSocket.rc = comp.rc;
                    body.rightTopSocket.id = comp.id;
                } else if (comp.id == COMPONENT_RBSOCKET) {
                    body.rightBottomSocket.image = comp.elements[0];
                    body.rightBottomSocket.text = comp.text;
                    body.rightBottomSocket.rc = comp.rc;
                    body.rightBottomSocket.id = comp.id;
                } else if (comp.id == COMPONENT_PARAMETERS) {
					//TODO
				}
			}
			param.body = body;
		}
	}
	return param;
}

Component ZtfUtil::_parseComponent(rapidxml::xml_node<>* node)
{
	Component comp;
    //auto child2 = node->first_node();
	auto child = node->first_node("text");
	comp.text = _parseText(child);
	comp.id = node->first_attribute("id")->value();
    qreal x, y, w, h;
	if (auto attr = node->first_attribute("x"))		x = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("y"))		y = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("w"))		w = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("h"))		h = QString(attr->value()).toFloat();
    comp.rc = QRect(x, y, w, h);
	for (auto child = node->first_node("image"); child != nullptr; child = child->next_sibling())
	{
		comp.elements.append(_parseImage(child));
	}
	return comp;
}

TextElement ZtfUtil::_parseText(rapidxml::xml_node<>* node)
{
	TextElement elem;
	if (!node)
		return elem;

    qreal x, y, w, h;
    QString fontFamily;
	if (auto attr = node->first_attribute("id")) elem.id = QString(attr->value());
	if (auto attr = node->first_attribute("x"))	x = QString(attr->value()).toInt();
    if (auto attr = node->first_attribute("y")) y = QString(attr->value()).toInt();
    if (auto attr = node->first_attribute("w")) w = QString(attr->value()).toFloat();
    if (auto attr = node->first_attribute("h")) h = QString(attr->value()).toFloat();
    if (auto attr = node->first_attribute("font-family"))
        fontFamily = QString(attr->value());
    elem.rc = QRectF(x, y, w, h);
	//attr = node->first_attribute("font-family");
	//attr = node->first_attribute("font-color");
	//attr = node->first_attribute("font-size");
	elem.text = node->value();
    elem.font = QFont(fontFamily);
	return elem;
}

ImageElement ZtfUtil::_parseImage(rapidxml::xml_node<>* node)
{
	ImageElement elem;
	if (!node)
		return elem;

	if (auto attr = node->first_attribute("id"))	elem.id = QString(attr->value());
	if (auto attr = node->first_attribute("normal"))	elem.image = QString(attr->value());
	if (auto attr = node->first_attribute("selected"))	elem.imageOn = QString(attr->value());
    if (auto attr = node->first_attribute("hovered"))	elem.imageHovered = QString(attr->value());

	qreal x = 0, y = 0, w = 0, h = 0;

	if (auto attr = node->first_attribute("x"))		x = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("y"))		y = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("w"))		w = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("h"))		h = QString(attr->value()).toFloat();

	elem.rc = QRectF(x, y, w, h);

	return elem;
}



const char* ZtfUtil::qsToString(const QString &qs)
{
    std::string s = qs.toStdString();
    char *wtf = new char[s.size() + 1];
    strcpy(wtf, s.c_str());
	return wtf;
}

void ZtfUtil::_exportRc(rapidxml::xml_document<> &doc, XML_NODE node, QRect rc)
{
    node->append_attribute(doc.allocate_attribute("x", qsToString(QString::number(rc.left()))));
    node->append_attribute(doc.allocate_attribute("y", qsToString(QString::number(rc.top()))));
    node->append_attribute(doc.allocate_attribute("w", qsToString(QString::number(rc.width()))));
    node->append_attribute(doc.allocate_attribute("h", qsToString(QString::number(rc.height()))));
}
		
void ZtfUtil::_exportRcF(rapidxml::xml_document<> &doc, XML_NODE node, QRectF rc)
{
    node->append_attribute(doc.allocate_attribute("x", qsToString(QString::number(rc.left()))));
    node->append_attribute(doc.allocate_attribute("y", qsToString(QString::number(rc.top()))));
    node->append_attribute(doc.allocate_attribute("w", qsToString(QString::number(rc.width()))));
    node->append_attribute(doc.allocate_attribute("h", qsToString(QString::number(rc.height()))));
}

XML_NODE ZtfUtil::_exportImage(XMLDOC_REF doc, ImageElement imageElem)
{
    XML_NODE imgNode = doc.allocate_node(rapidxml::node_element, "image", "");
    imgNode->append_attribute(doc.allocate_attribute("id", qsToString(imageElem.id)));
    _exportRcF(doc, imgNode, imageElem.rc);
    imgNode->append_attribute(doc.allocate_attribute("normal", qsToString(imageElem.image)));
    imgNode->append_attribute(doc.allocate_attribute("hovered", qsToString(imageElem.imageHovered)));
    imgNode->append_attribute(doc.allocate_attribute("selected", qsToString(imageElem.imageOn)));
    return imgNode;
}

XML_NODE ZtfUtil::_exportText(XMLDOC_REF doc, TextElement textElem)
{
    XML_NODE textNode = doc.allocate_node(rapidxml::node_element, "text", qsToString(textElem.text));
    textNode->append_attribute(doc.allocate_attribute("id", qsToString(textElem.id)));
    _exportRcF(doc, textNode, textElem.rc);
    textNode->append_attribute(doc.allocate_attribute("font-family", qsToString(textElem.font.family())));
    //nameElem->append_attribute(doc.allocate_attribute("font-color", qsToString(textElem.font.f)));
    return textNode;
}

XML_NODE ZtfUtil::_exportHeader(rapidxml::xml_document<> &doc, HeaderParam headerParam)
{
    XML_NODE header = doc.allocate_node(rapidxml::node_element, "header", "");
	//node-name
    XML_NODE node_name = doc.allocate_node(rapidxml::node_element, "component", "");
    node_name->append_attribute(doc.allocate_attribute("id", qsToString(headerParam.name.id)));
    _exportRc(doc, node_name, headerParam.name.rc);
	{
        XML_NODE nameElem = _exportText(doc, headerParam.name.text);
		node_name->append_node(nameElem);
	}

    XML_NODE status = doc.allocate_node(rapidxml::node_element, "component", "");
    status->append_attribute(doc.allocate_attribute("id", qsToString(headerParam.status.id)));
    _exportRc(doc, status, headerParam.status.rc);
	{
        status->append_node(_exportImage(doc, headerParam.status.mute));
        status->append_node(_exportImage(doc, headerParam.status.view));
        status->append_node(_exportImage(doc, headerParam.status.prep));
	}

	XML_NODE control = doc.allocate_node(rapidxml::node_element, "component", "");
    control->append_attribute(doc.allocate_attribute("id", qsToString(headerParam.control.id)));
    _exportRc(doc, control, headerParam.control.rc);
    {
        control->append_node(_exportImage(doc, headerParam.control.elements[0]));
    }

	XML_NODE display = doc.allocate_node(rapidxml::node_element, "component", "");
    display->append_attribute(doc.allocate_attribute("id", qsToString(headerParam.display.id)));
    _exportRc(doc, display, headerParam.display.rc);
    {
        display->append_node(_exportImage(doc, headerParam.display.image));
    }

    XML_NODE header_bg = doc.allocate_node(rapidxml::node_element, "component", "");
    header_bg->append_attribute(doc.allocate_attribute("id", qsToString(headerParam.backboard.id)));
    _exportRc(doc, header_bg, headerParam.backboard.rc);
    {
        header_bg->append_node(_exportImage(doc, headerParam.backboard.image));
    }

	header->append_node(node_name);
    header->append_node(status);
    header->append_node(control);
    header->append_node(display);
    header->append_node(header_bg);

	return header;
}

XML_NODE ZtfUtil::_exportBody(rapidxml::xml_document<> &doc, BodyParam bodyParam)
{
    XML_NODE body = doc.allocate_node(rapidxml::node_element, "body", "");

    XML_NODE ltsocket = doc.allocate_node(rapidxml::node_element, "component", "");
    ltsocket->append_attribute(doc.allocate_attribute("id", qsToString(bodyParam.leftTopSocket.id)));
    _exportRc(doc, ltsocket, bodyParam.leftTopSocket.rc);
    {
        ltsocket->append_node(_exportImage(doc, bodyParam.leftTopSocket.image));
        ltsocket->append_node(_exportText(doc, bodyParam.leftTopSocket.text));
    }

    XML_NODE lbsocket = doc.allocate_node(rapidxml::node_element, "component", "");
    lbsocket->append_attribute(doc.allocate_attribute("id", qsToString(bodyParam.leftBottomSocket.id)));
    _exportRc(doc, lbsocket, bodyParam.leftBottomSocket.rc);
    {
        lbsocket->append_node(_exportImage(doc, bodyParam.leftBottomSocket.image));
        lbsocket->append_node(_exportText(doc, bodyParam.leftBottomSocket.text));
    }

    XML_NODE rtsocket = doc.allocate_node(rapidxml::node_element, "component", "");
    rtsocket->append_attribute(doc.allocate_attribute("id", qsToString(bodyParam.rightTopSocket.id)));
    _exportRc(doc, rtsocket, bodyParam.rightTopSocket.rc);
    {
        rtsocket->append_node(_exportImage(doc, bodyParam.rightTopSocket.image));
        rtsocket->append_node(_exportText(doc, bodyParam.rightTopSocket.text));
    }

    XML_NODE rbsocket = doc.allocate_node(rapidxml::node_element, "component", "");
    rbsocket->append_attribute(doc.allocate_attribute("id", qsToString(bodyParam.rightBottomSocket.id)));
    _exportRc(doc, rbsocket, bodyParam.rightBottomSocket.rc);
    {
        rbsocket->append_node(_exportImage(doc, bodyParam.rightBottomSocket.image));
        rbsocket->append_node(_exportText(doc, bodyParam.rightBottomSocket.text));
    }

    XML_NODE body_bg = doc.allocate_node(rapidxml::node_element, "component", "");
    body_bg->append_attribute(doc.allocate_attribute("id", qsToString(bodyParam.backboard.id)));
    _exportRc(doc, body_bg, bodyParam.backboard.rc);
    {
        body_bg->append_node(_exportImage(doc, bodyParam.backboard.image));
    }

    //TODO: paramters

    body->append_node(ltsocket);
    body->append_node(lbsocket);
    body->append_node(rtsocket);
    body->append_node(rbsocket);
    body->append_node(body_bg);
	return body;
}

void ZtfUtil::exportZtf(NodeParam param, const QString &output)
{
    rapidxml::xml_document<> doc;
    XML_NODE node = doc.allocate_node(rapidxml::node_element, "node", "");
    doc.append_node(node);

    node->append_node(_exportHeader(doc, param.header));
    node->append_node(_exportBody(doc, param.body));

    std::string s;
    print(std::back_inserter(s), doc, 0);

    QString qsXml = QString::fromStdString(s);

	QFile f(output);
	f.open(QIODevice::WriteOnly);
    f.write(qsXml.toUtf8());
	f.close();
}