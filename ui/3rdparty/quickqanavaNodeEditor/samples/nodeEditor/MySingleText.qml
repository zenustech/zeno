import QtQuick                   2.3
import QtQuick.Controls          2.1
Item{
    property alias txt:textInput.text
    property alias backGroundColor:root.color
    property var textColor: "#FFFFFF"

    Rectangle {
        id:root
        anchors.fill: parent

        TextInput{
            id : textInput
            anchors.fill: parent
            //anchors.margins : root.margins
            font.family : "Ubuntu Mono, Courier New, Courier"
            font.pixelSize: 14
            font.weight : Font.Normal
            text: 'this is single intput'
            color: textColor
            selectByMouse: true
            //selectionColor: root.textSelectionColor
            clip:true

            property bool touched : false

            Keys.onPressed : root.keyPressed(event)

            MouseArea{
                anchors.fill: parent
                acceptedButtons: Qt.NoButton
                cursorShape: Qt.IBeamCursor
            }
        }

        Text {
            anchors.fill: parent
            anchors.margins : textInput.anchors.margins
            //text: root.textHint
            font: textInput.font
            //color: root.hintTextColor
            visible: !textInput.text && !textInput.activeFocus
        }
    }
}