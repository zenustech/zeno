import QtQuick 2.15
import QtQuick.Controls 2.2
import QtQuick.Shapes 1.15

Item {
    id: comp
    property int xoffset: 14
    property int side: 35
    property bool lastBtn: false

    //property bool checked: false
    //property bool hovered: false  //外部定义已经alias

    implicitWidth: width
    implicitHeight: height

    property string source: ""
    property string source_on: ""
    property alias mouseArea: mouseArea

    Image{
        id: img
        sourceSize.width: comp.width
        sourceSize.height: comp.height
        smooth: true
        antialiasing: true
        source: (comp.checked || comp.hovered) ? comp.source_on : comp.source
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
            onClicked: {
                comp.checked = !comp.checked
                comp.hovered = false
            }
            onEntered: {
                comp.hovered = true
            }
            onExited: {
                comp.hovered = false
            }
        }
    }
}
