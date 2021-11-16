#ifndef __ZTF_UTIL_H__
#define __ZTF_UTIL_H__

#include "renderparam.h"
#include <rapidxml/rapidxml_print.hpp>

class ZtfUtil
{
public:
	static ZtfUtil& GetInstance();
	NodeParam loadZtf(const QString& filename);

private:
	ZtfUtil();
	Component _parseComponent(rapidxml::xml_node<>*);
	TextElement _parseText(rapidxml::xml_node<>*);
	ImageElement _parseImage(rapidxml::xml_node<>*);
};

#endif