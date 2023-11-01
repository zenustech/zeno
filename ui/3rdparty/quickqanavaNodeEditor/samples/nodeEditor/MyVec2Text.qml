import QtQuick                   2.3
import QtQuick.Controls          2.1
import QtQuick.Layouts           1.3

Item{
    property alias txt:textInput.text
    property alias backGroundColor:root.bgClr
    property var textColor: "#FFFFFF"

    RowLayout {
        id:root
        //color:"lightblue"
        anchors.fill:parent
        layoutDirection: "LeftToRight"
        //spacing:10
        property var bgClr;

        Rectangle{
            Layout.fillWidth: parent.width
            Layout.fillHeight: parent.height
            color: parent.bgClr

            TextInput{
                id : textInput
                anchors.fill: parent
                //anchors.margins : root.margins
                font.family : "Ubuntu Mono, Courier New, Courier"
                font.pixelSize: 14
                font.weight : Font.Normal
                text: 'this is single intput'
                color : textColor
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
        }
        Rectangle{
            Layout.fillWidth: parent.width
            Layout.fillHeight: parent.height
            color: parent.bgClr

            TextInput{
                id : textInput2
                anchors.fill: parent
                //anchors.margins : root.margins
                font.family : "Ubuntu Mono, Courier New, Courier"
                font.pixelSize: 14
                font.weight : Font.Normal
                text: 'this is single intput'
                color : textColor
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
        }
    }
}