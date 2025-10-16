import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Controls.Styles   1.4
import zeno.enum 1.0


Item {
    id: comp

    property string socket_name;
    property int socket_group;
    property color bg_color;

    signal socketClicked()

    height: mainLayout.implicitHeight
    width: mainLayout.implicitWidth

    RowLayout {
        id: mainLayout
        z: 100
        //Layout.margins: 2
        Text {
            Layout.topMargin: 2
            Layout.leftMargin: 8
            Layout.rightMargin: 8
            Layout.bottomMargin: 2
            color: "white"
            text: comp.socket_name
            font.pixelSize: 14
        }
    }

    Rectangle {
        anchors.fill: parent
        color: comp.bg_color
        radius: 4
    }

    MouseArea {
        id: comp_mousearea
        anchors.fill: parent

        onClicked: {
            comp.socketClicked()
        }
    }
}