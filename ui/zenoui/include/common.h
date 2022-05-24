#ifndef __ZENO_UI_COMMON_DEF_H__
#define __ZENO_UI_COMMON_DEF_H__

enum SearchType
{
	SEARCH_SUBNET = 1 << 0,
	SEARCH_NODECLS = 1 << 1,
	SEARCH_NODEID = 1 << 2,		// search node ident.
	SEARCH_ANNO = 1 << 3,
	SEARCH_WRANGLE = 1 << 4,
	SEARCHALL = SEARCH_NODECLS | SEARCH_NODEID | SEARCH_SUBNET | SEARCH_ANNO | SEARCH_WRANGLE
};

struct SEARCH_RESULT
{
	SearchType type;
	QModelIndex targetIdx;
	QModelIndex subgIdx;
};


#endif