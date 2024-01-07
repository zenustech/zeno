import QtQuick 2.15
import QtQuick.Controls 2.2
import QtQuick.Shapes 1.15

Item {
    id: comp
    property int xoffset: 14
    property int side: 35
    property bool lastBtn: false

    property bool clicked: false

    implicitWidth: img.width
    implicitHeight: img.height

    signal clickedSig()
    signal enteredSig()
    signal exitedSig()

    property alias source: img.source
    property alias mouseArea: mouseArea

    Image{
        id: img
    }
    Shape {
        id: sp
        anchors.fill: parent
        antialiasing: true

        containsMode: Shape.FillContains
        ShapePath {
            id: path
            strokeColor: "transparent"
            fillColor: "transparent"

            startX: comp.xoffset
            startY: 0
            PathLine { x: comp.xoffset + comp.side - (comp.lastBtn ? comp.xoffset : 0) + 3; y: 0 }
            PathLine { x: comp.xoffset + comp.side - comp.xoffset + 3; y: comp.height}
            PathLine { x: 0; y: comp.height }
            PathLine { x: comp.xoffset; y: 0 }
        }
        MouseArea {
            id: mouseArea
            anchors.fill: parent
            containmentMask: parent
            hoverEnabled: true
            acceptedButtons: Qt.LeftButton
            onClicked: clickedSig()
            onEntered: enteredSig()
            onExited: exitedSig()
        }
    }
}
