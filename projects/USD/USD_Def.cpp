#include<zeno/zeno.h>
#include<USD.h>
ZENDEFNODE(ImportUSDMesh,
    {
        /* inputs */
        {
            {"readpath", "usdPath", ""},
            {"string", "primPath", ""},
            {"float", "frame", "-1"}
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
            {"readpath", "usdPath", ""},
            {"string", "primPath", ""},
            {"string", "opName", ""},
            {"float", "frame", "-1"},
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
            {"readpath", "usdPath", ""}
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
        {"readpath", "usdPath", ""}
    },
/* outputs */
{
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
            {"readpath", "usdPath", ""},
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
            {"readpath", "usdPath", ""},
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
            {"readpath", "usdPath", ""},
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