#ifndef __ZENO_UI_COMMON_DEF_H__
#define __ZENO_UI_COMMON_DEF_H__

enum SearchType
{
	SEARCH_SUBNET = 1 << 0,
	SEARCH_NODE = 1 << 1,
	SEARCH_ANNO = 1 << 2,
	SEARCH_WRANGLE = 1 << 3,
	SEARCHALL = SEARCH_NODE | SEARCH_SUBNET | SEARCH_ANNO | SEARCH_WRANGLE
};

struct SEARCH_RESULT
{
	SearchType type;
	QModelIndex targetIdx;
	QModelIndex subgIdx;
};


#endif