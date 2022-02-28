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
                QString componentName = component->name();
                if (componentName == "background")
                {
                    BackgroundComponent comp = _parseBackground(component);
                    header.backboard = comp;
                }
                else
                {
                    Component comp = _parseComponent(component);
                    if (comp.id == COMPONENT_STATUS) {
                        header.status.id = comp.id;
                        header.status.mute = comp.elements[0];
                        header.status.view = comp.elements[1];
                        header.status.once = comp.elements[2];
                        header.status.rc = comp.rc;
                    } else if (comp.id == COMPONENT_CONTROL) {
                        header.control = comp;
                    } else if (comp.id == COMPONENT_DISPLAY) {
                        header.display.id = comp.id;
                        header.display.image = comp.elements[0];
                        header.display.rc = comp.rc;
                    } else if (comp.id == COMPONENT_NAME) {
                        header.name.id = comp.id;
                        header.name.text = comp.text;
                        header.name.rc = comp.rc;
                    }
                }
			}
			param.header = header;
		}
		else if (name == "body")
		{
			BodyParam body;
			for (auto component = child->first_node(); component != nullptr; component = component->next_sibling())
			{
                QString componentName = component->name();
                if (componentName == "background")
                {
                    BackgroundComponent comp = _parseBackground(component);
                    body.backboard = comp;
                }
                else
                {
                    Component comp = _parseComponent(component);
                    if (comp.id == COMPONENT_LTSOCKET) {
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

BackgroundComponent ZtfUtil::_parseBackground(rapidxml::xml_node<>* node)
{
    BackgroundComponent comp;
    if (!node)
        return comp;

    comp.bApplyImage = false;
    qreal x, y, w, h;
    if (auto attr = node->first_attribute("id")) comp.id = QString(attr->value());
    if (auto attr = node->first_attribute("x")) x = QString(attr->value()).toInt();
    if (auto attr = node->first_attribute("y")) y = QString(attr->value()).toInt();
    if (auto attr = node->first_attribute("w")) w = QString(attr->value()).toFloat();
    if (auto attr = node->first_attribute("h")) h = QString(attr->value()).toFloat();
    comp.rc = QRect(x, y, w, h);

    if (auto attr = node->first_attribute("radius"))
    {
        QStringList rxs = QString(attr->value()).split(" ");
        if (rxs.size() == 4)
        {
            comp.lt_radius = rxs[0].mid(0, rxs[0].indexOf("px")).toInt();
            comp.rt_radius = rxs[1].mid(0, rxs[1].indexOf("px")).toInt();
            comp.lb_radius = rxs[2].mid(0, rxs[2].indexOf("px")).toInt();
            comp.rb_radius = rxs[3].mid(0, rxs[3].indexOf("px")).toInt();
        }
    }
    if (auto attr = node->first_attribute("normal-clr"))
    {
        comp.clr_normal = QColor(attr->value());
    }
    if (auto attr = node->first_attribute("hoverd-clr"))
    {
        QString val(attr->value());
        comp.bAcceptHovers = !val.isEmpty();
        comp.clr_hovered = QColor(val);
    }
    if (auto attr = node->first_attribute("selected-clr"))
    {
        comp.clr_selected = QColor(attr->value());
    }
    if (auto attr = node->first_attribute("border-clr"))
    {
        comp.clr_border = QColor(attr->value());
    }
    if (auto attr = node->first_attribute("border-width"))
    {
         QString bw = QString(attr->value());
         comp.border_witdh = bw.mid(0, bw.indexOf("px", 0, Qt::CaseInsensitive)).toInt();
    }
    //todo: gradient

    auto img = node->first_node("image");
    comp.imageElem = _parseImage(img);
    if (!comp.imageElem.image.isEmpty())
    {
        comp.bApplyImage = true;
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

XML_NODE ZtfUtil::_exportBackground(XMLDOC_REF doc, BackgroundComponent bg)
{
    XML_NODE node = doc.allocate_node(rapidxml::node_element, "background", "");
    node->append_attribute(doc.allocate_attribute("id", qsToString(bg.id)));
    _exportRc(doc, node, bg.rc);

    node->append_attribute(doc.allocate_attribute("radius", qsToString(QString("%1px %2px %3px %4px").arg(bg.lt_radius).arg(bg.rt_radius).arg(bg.lb_radius).arg(bg.rb_radius))));
    node->append_attribute(doc.allocate_attribute("normal-clr", qsToString(bg.clr_normal.name())));
    node->append_attribute(doc.allocate_attribute("hoverd-clr", qsToString(bg.clr_hovered.name())));
    node->append_attribute(doc.allocate_attribute("selected-clr", qsToString(bg.clr_selected.name())));
    {
        node->append_node(_exportImage(doc, bg.imageElem));
    }
    return node;
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
        status->append_node(_exportImage(doc, headerParam.status.once));
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

    XML_NODE header_bg = _exportBackground(doc, headerParam.backboard);

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

    XML_NODE body_bg = _exportBackground(doc, bodyParam.backboard);

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


NodeUtilParam ZtfUtil::toUtilParam(const NodeParam& nodeParam)
{
    NodeUtilParam param;

    QPoint base = nodeParam.header.backboard.rc.topLeft();

    //header
    param.headerBg = nodeParam.header.backboard;
    param.headerBg.rc = nodeParam.header.backboard.rc.translated(-base);

    param.mute = nodeParam.header.status.mute;
    param.view = nodeParam.header.status.view;
    param.prep = nodeParam.header.status.once;
    param.rcMute = nodeParam.header.status.mute.rc.translated(-base);
    param.rcView = nodeParam.header.status.view.rc.translated(-base);
    param.rcPrep = nodeParam.header.status.once.rc.translated(-base);
    param.status = nodeParam.header.status;

    param.collaspe = nodeParam.header.control.elements[0];
    param.rcCollasped = nodeParam.header.control.elements[0].rc.translated(-base);

    param.name = nodeParam.header.name.text;
    param.namePos = param.name.rc.topLeft() - base;

    //body
    param.bodyBg = nodeParam.body.backboard;
    param.bodyBg.rc = nodeParam.body.backboard.rc.translated(-base);

    param.socket = nodeParam.body.leftTopSocket.image;
    param.szSocket = QSizeF(nodeParam.body.leftTopSocket.image.rc.width(), nodeParam.body.leftTopSocket.image.rc.height());

    param.socketHOffset = std::abs(nodeParam.body.leftTopSocket.image.rc.left() - 
                        nodeParam.body.backboard.rc.left());
    param.socketToText = std::abs(nodeParam.body.leftTopSocket.image.rc.right() -
                                  nodeParam.body.leftTopSocket.text.rc.left());
    param.socketVMargin = std::abs(nodeParam.body.leftTopSocket.image.rc.bottom() -
                                  nodeParam.body.leftBottomSocket.image.rc.top());
    //todo: parameterized.
    param.nameFont = QFont("HarmonyOS Sans", 13);
    param.nameFont.setBold(true);
    param.socketFont = QFont("HarmonyOS Sans", 11);
    param.socketFont.setBold(true);
    param.paramFont = QFont("HarmonyOS Sans", 11);
    param.paramFont.setBold(true);

    QColor clr(255, 255, 255);
    clr.setAlphaF(0.4);
    param.nameClr = clr;
    param.socketClr = QColor(188, 188, 188);
    clr = QColor(255, 255, 255);
    clr.setAlphaF(0.7);
    param.paramClr = clr;

    param.boardFont = QFont("Consolas", 17);
    param.boardTextClr = QColor(255, 255, 255);

    param.lineEditParam.font = QFont("HarmonyOS Sans SC", 10);
    QPalette palette;
    palette.setColor(QPalette::Base, QColor(37, 37, 37));
    //palette.setColor(QPalette::Active, QPalette::WindowText, QColor(228, 228, 228));
    //palette.setColor(QPalette::Inactive, QPalette::WindowText, QColor(158, 158, 158));
    clr = QColor(255, 255, 255);
    clr.setAlphaF(0.4);
    palette.setColor(QPalette::Text, clr);
    param.lineEditParam.palette = palette;
    param.lineEditParam.margins = QMargins(8, 0, 0, 0);

    param.comboboxParam.font = QFont("HarmonyOS Sans SC", 13);
    param.comboboxParam.textColor = QColor(255, 255, 255);
    param.comboboxParam.itemBgNormal = QColor(33, 33, 33);
    param.comboboxParam.itemBgHovered = QColor(23, 160, 252);
    param.comboboxParam.itemBgSelected = QColor(23, 160, 252);
    param.comboboxParam.margins = QMargins(8, 0, 0, 0);
    palette.setColor(QPalette::Base, QColor(37, 37, 37));
    palette.setColor(QPalette::Window, QColor(37, 37, 37));
    palette.setColor(QPalette::Text, Qt::white);
    palette.setColor(QPalette::Active, QPalette::WindowText, QColor());
    palette.setColor(QPalette::Inactive, QPalette::WindowText, QColor());

    param.comboboxParam.palette = palette;

    return param;
}