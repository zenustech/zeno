import QtQuick 2.12
import QtQuick.Controls 2.0

Item{
    width: 64
    height: 26

    property alias text: textInput.text

    TextField{
        id : textInput
        //anchors.margins: 0
        anchors.fill: parent
        //verticalAlignment: TextInput.AlignVCenter
        clip:true
        padding: 0

        color: "#FFF"
        selectionColor: "#0078D7"
        font.pointSize: 12
        font.family: "Consolas"

        focus: true
        selectByMouse: true
        Keys.onEscapePressed: focus = false

        background:Rectangle {
            id: backGround
            color: "#191D21"
            //border.color: "grey"
        }
    }
}