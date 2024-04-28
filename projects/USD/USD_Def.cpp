#include<zeno/zeno.h>
#include<USD.h>
ZENDEFNODE(ReadUSD,
    {
        /* inputs */
        {
            {"readpath", "path", ""}
        },
        /* outputs */
        {
            {"string", "USDDescription"}
        },
        /* params */
        {},
        /* category */
        {"USD"}
    }
);
ZENDEFNODE(ImportUSDMesh,
    {
        /* inputs */
        {
            {"string", "USDDescription", ""},
            {"string", "primPath", ""}
        },
        /* outputs */
        {
            {"primitive", "prim"}
        },
        /* params */
        {},
        /* category */
        {"USD"}
    }
);
ZENDEFNODE(ImportUSDPrimMatrix,
    {
        /* inputs */
        {
            {"string", "USDDescription", ""},
            {"string", "primPath", ""},
            {"string", "opName", ""},
            {"bool", "isInversedOp", "0"},
        },
    /* outputs */
    {
        {"Matrix"}
    },
    /* params */
    {},
    /* category */
    {"USD"}
    }
);
ZENDEFNODE(ViewUSDTree,
    {
        /* inputs */
        {
            {"string", "USDDescription", ""}
        },
    /* outputs */
    {
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });
ZENDEFNODE(USDShowAllPrims,
{
    /* inputs */
    {
        {"string", "USDDescription", ""}
    },
/* outputs */
{
},
/* params */
{},
/* category */
{"USD"}
});
ZENDEFNODE(ShowPrimUserData,
    {
    /* inputs */
    {
        {"primitive", "prim", ""},
    },
    /* outputs */
    {
        // {"primitive", "prim"}
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });
ZENDEFNODE(ShowUSDPrimAttribute,
    {
        /* inputs */
        {
            {"string", "USDDescription", ""},
            {"string", "primPath", ""},
            {"string", "attributeName", ""}
        },
        /* outputs */
        {
            // {"primitive", "prim"}
        },
        /* params */
        {},
        /* category */
        {"USD"}
    });

ZENDEFNODE(ShowUSDPrimRelationShip,
    {
        /* inputs */
        {
            {"string", "USDDescription", ""},
            {"string", "primPath", ""}
        },
    /* outputs */
    {
        // {"primitive", "prim"}
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });

ZENDEFNODE(EvalUSDPrim,
    {
        /* inputs */
        {
            {"readpath", "USDDescription", ""},
            {"string", "primPath", ""},
            {"bool", "isRecursive", "0"},
        },
        /* outputs */
        {
            // {"primitive", "prim"}
        },
        /* params */
        {},
        /* category */
        {"USD"}
    }
);