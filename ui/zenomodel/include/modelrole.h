#ifndef __MODEL_ROLE_H__
#define __MODEL_ROLE_H__

enum MODEL_ROLE
{
	ROLE_OBJID = Qt::UserRole + 1,	// like "0xgrwe-MakeCurveMap", which is a ident.
	ROLE_OBJNAME,	// like "MakeCurveMap", which is a node "class" name.
	ROLE_NODETYPE,
	ROLE_OBJPOS,
	ROLE_OBJRECT,
	ROLE_OBJPATH,	// unique path for subgraph, node and param, even curvemodel.
	ROLE_OBJDATA,
	ROLE_LINK_IDX,	// link index
	ROLE_NODE_IDX,
	ROLE_SUBGRAPH_IDX,	// subgraph index.
	ROLE_PARAMETERS,
	ROLE_INPUTS,
	ROLE_PARAMS_NO_DESC,
	ROLE_OUTPUTS,
	ROLE_OPTIONS,
	ROLE_COLLASPED,
	ROLE_INPUT_PARAM,		//input param model index.
	ROLE_OUTPUT_PARAM,		//output param model index.

	ROLE_INPUT_MODEL,		//input socket model
	ROLE_PARAM_MODEL,		//param model
	ROLE_OUTPUT_MODEL,		//output socket model.
	ROLE_PANEL_PARAMS,	// all viewed panel parameters.
	ROLE_CUSTOMUI_PANEL_IO,	// temp storage for custom panel io. see VARAM_INFO
	ROLE_NODE_PARAMS,		// all viewed node parameters.

	//synchronize link change, param
	ROLE_ADDLINK,
	ROLE_REMOVELINK,
	ROLE_MODIFY_PARAM,
	ROLE_MODIFY_SOCKET,
	ROLE_MODIFY_SOCKET_DEFL,

	//link role
	ROLE_OUTNODE,
	ROLE_INNODE,
	ROLE_OUTSOCK,
	ROLE_INSOCK,
	ROLE_INNODE_IDX,
	ROLE_OUTNODE_IDX,
	ROLE_OUTSOCK_IDX,
	ROLE_INSOCK_IDX,

	//parameter model role
	ROLE_PARAM_NAME,
	ROLE_PARAM_TYPE,
	ROLE_PARAM_CTRL,		//ui control
	ROLE_PARAM_VALUE,
	ROLE_PARAM_LINKS,
	ROLE_PARAM_SOCKPROP,	//socket property, see SOCKET_PROPERTY.
	ROLE_PARAM_CLASS,	//just tell whether the param is input socket, outputsocket or param, see PARAM_CLASS
	ROLE_PARAM_COREIDX,		//return the core param idx under view param.

	//view param
	ROLE_VPARAM_TYPE,		//vtype, such as group tab param, not ROLE_PARAM_TYPE
    ROLE_VPARAM_NAME,
    ROLE_VPARAM_IS_COREPARAM,   //is mapped from core param.
    ROLE_VAPRAM_EDITTABLE,       //edittable for name and content.
    ROLE_VPARAM_ACTIVE_TABINDEX,    //active tab index
    ROLE_VPARAM_COLLASPED,      // whether group is collasped.
	ROLE_VPARAM_CTRL_PROPERTIES,
	ROLE_VPARAM_LINK_MODEL,     // a qstandarditem model to represent the collection of the links for a socket.
    ROLE_VPARAM_TOOLTIP,
};

enum LOG_ROLE
{
    ROLE_LOGTYPE = Qt::UserRole + 1,
	ROLE_TIME,
	ROLE_FILENAME,
	ROLE_LINENO,
	ROLE_NODE_IDENT,
	ROLE_RANGE_START,
	ROLE_RANGE_LEN
};

#endif
