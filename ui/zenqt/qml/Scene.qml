import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import ZQuickParam 1.0


//Descriptor, 参照zsg的描述格式:
var nodedescs = {
    "CreateCube" : {
        "inputs": [
             {
                 "name": "position",
                 "type": "vec3f",
                 "control": ZQuickParam.CTRL_VEC3F,
                 "value": [0,0,0]
             },
             {
                 "name": "scale",
                 "type": "vec3f",
                 "control": ZQuickParam.CTRL_VEC3F,
                 "value": [0,1,0]
             }
        ],
        "outputs": [
             {
                 "name": "prim",
                 "type": "prim"
             }
        ]
    },
    "GetFrameNum" : {
        "inputs": [
            {
                "name": "SRC"
            }
        ],
        "outputs": [
            {
                "name": "FrameNum",
                "type": "int"
            },
            {
                "name": "DST"
            }
        ]
    }

}



Item {
    id: scene



}