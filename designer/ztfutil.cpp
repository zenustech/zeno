#include "ztfutil.h"

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
	Q_ASSERT(ret);
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
				if (comp.id == "status")
				{
					header.status = comp;
				}
				else if (comp.id == "control")
				{
					header.control = comp;
				}
				else if (comp.id == "display")
				{
					header.display = comp;
				}
				else if (comp.id == "backboard")
				{
					header.backborad = comp;
				}
				else if (comp.id == "name")
				{
					header.name = comp;
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
				if (comp.id == "backboard")
				{
					body.backboard = comp;
				}
				else if (comp.id == "topleftsocket")
				{
					body.leftTopSocket = comp;
				}
				else if (comp.id == "bottomleftsocket")
				{
					body.leftBottomSocket = comp;
				}
				else if (comp.id == "toprightsocket")
				{
					body.rightTopSocket = comp;
				}
				else if (comp.id == "bottomrightsocket")
				{
					body.rightBottomSocket = comp;
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
	auto child = node->first_node("text");
	comp.text = _parseText(child);
	comp.id = node->first_attribute("id")->value();
	if (auto attr = node->first_attribute("x"))		comp.x = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("y"))		comp.y = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("w"))		comp.w = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("h"))		comp.h = QString(attr->value()).toFloat();
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

	if (auto attr = node->first_attribute("x"))	elem.x = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("y"))	elem.y = QString(attr->value()).toFloat();
	//attr = node->first_attribute("font-family");
	//attr = node->first_attribute("font-color");
	//attr = node->first_attribute("font-size");
	elem.text = node->value();
	return elem;
}

ImageElement ZtfUtil::_parseImage(rapidxml::xml_node<>* node)
{
	ImageElement elem;
	if (!node)
		return elem;

	if (auto attr = node->first_attribute("id"))	elem.id = QString(attr->value());
	if (auto attr = node->first_attribute("type"))	elem.type = QString(attr->value());
	if (auto attr = node->first_attribute("normal"))	elem.image = QString(attr->value());
	if (auto attr = node->first_attribute("selected"))	elem.imageOn = QString(attr->value());
	if (auto attr = node->first_attribute("x"))		elem.x = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("y"))		elem.y = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("w"))		elem.w = QString(attr->value()).toFloat();
	if (auto attr = node->first_attribute("h"))		elem.h = QString(attr->value()).toFloat();
	return elem;
}