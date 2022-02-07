#ifndef __DESIGNER_COMMON_ID_H__
#define __DESIGNER_COMMON_ID_H__

#define COMPONENT_NAME "name"
#define COMPONENT_STATUS "status"
#define COMPONENT_CONTROL "control"
#define COMPONENT_DISPLAY "display"
#define COMPONENT_HEADER_BG "header-backboard"
#define COMPONENT_LTSOCKET "topleftsocket"
#define COMPONENT_LBSOCKET "bottomleftsocket"
#define COMPONENT_RTSOCKET "toprightsocket"
#define COMPONENT_RBSOCKET "bottomrightsocket"
#define COMPONENT_BODY_BG   "body-backboard"
#define COMPONENT_PARAMETERS "parameters"

#define ELEMENT_NAME "node-name"
#define ELEMENT_MUTE "mute"
#define ELEMENT_ONCE "once"
#define ELEMENT_VIEW "view"
#define ELEMENT_PREP "prep"
#define ELEMENT_COLLAPSE        "collapse"
#define ELEMENT_DISPLAY         "display-image"
#define ELEMENT_BODY_BG         "body-backboard-image"
#define ELEMENT_HEADER_BG       "header-backboard-image"
#define ELEMENT_LTSOCKET_IMAGE  "ltsocket-image"
#define ELEMENT_LTSOCKET_TEXT   "ltsocket-text"
#define ELEMENT_RTSOCKET_IMAGE  "rtsocket-image"
#define ELEMENT_RTSOCKET_TEXT   "rtsocket-text"
#define ELEMENT_LBSOCKET_IMAGE  "lbsocket-image"
#define ELEMENT_LBSOCKET_TEXT   "lbsocket-text"
#define ELEMENT_RBSOCKET_IMAGE  "rbsocket-image"
#define ELEMENT_RBSOCKET_TEXT   "rbsocket-text"

enum ZVALUE_ORDER
{
	ZVALUE_GRID_BACKGROUND,
	ZVALUE_GRID_SMALL,
	ZVALUE_GRID_BIG,
	ZVALUE_LINK = -8,	//numeric value influence the actual effect...
	ZVALUE_BLACKBOARD,
	ZVALUE_CORE_ITEM,
	ZVALUE_LOCKED_BG,
	ZVALUE_LOCKED_CP,
	ZVALUE_LOCKED_ELEM,
	ZVALUE_BACKGROUND = 0,
	ZVALUE_NODE_BORDER,
	ZVALUE_COMPONENT,
	ZVALUE_ELEMENT,
	ZVALUE_SELECTED,
};

#endif