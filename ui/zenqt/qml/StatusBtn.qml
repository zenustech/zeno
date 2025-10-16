import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Shapes 1.15


Item {
    id: comp
    property var basefillcolor
    property alias mouseAreaAlias: mouseArea

    property int xoffset: 12
    property int side: 24
    // property bool checked: false     //外部已定义alias
    property bool hovered: false
    property bool is_left_or_right: false        //是否为左边
    property bool round_last_btn: false  //底部是否使用圆弧
    property real radius: 4

    implicitWidth: xoffset + side
    implicitHeight: parent.height

    signal checkedChanged

    // layer.enabled: true
    // layer.samples: 64

    Shape {
        id: sp
        anchors.fill: parent
        antialiasing: true
        smooth: true

        containsMode: Shape.FillContains

        ShapePath {
            id: path
            strokeColor: "transparent"
            strokeWidth: 0

            fillColor: (comp.checked || comp.hovered) ? comp.basefillcolor : "#2E313A"
            capStyle: ShapePath.RoundCap
            
            property int joinStyleIndex: 0

            property variant styles: [
                ShapePath.BevelJoin,
                ShapePath.MiterJoin,
                ShapePath.RoundJoin
            ]

            property bool containsMouse: {
                //todo
                return true;
            }

            joinStyle: styles[joinStyleIndex]

            startX: comp.xoffset
            startY: 0
            PathLine { 
                x: {
                    if (comp.is_left_or_right) {
                        return comp.side - comp.radius;
                    }else{
                        return comp.xoffset + comp.side;
                    }
                }
                y: 0
            }
            PathArc {
                relativeX: comp.is_left_or_right ? comp.radius : 0
                relativeY: comp.is_left_or_right ? comp.radius : 0
                radiusX: comp.is_left_or_right ? comp.radius : 0
                radiusY: comp.is_left_or_right ? comp.radius : 0
                direction: PathArc.Clockwise
            }
            PathLine {
                x: comp.side
                y: {
                    if (comp.round_last_btn) {
                        return comp.height - comp.radius;
                    }
                    else{
                        return comp.height;
                    }
                }
            }
            PathArc {
                relativeX: comp.round_last_btn ? -comp.radius : 0
                relativeY: comp.round_last_btn ? comp.radius : 0
                radiusX: comp.round_last_btn ? comp.radius : 0
                radiusY: comp.round_last_btn ? comp.radius : 0
                direction: PathArc.Clockwise
            }
            PathLine { 
                x: 0;
                y: comp.height
            }
        }
        MouseArea {
            id: mouseArea
            anchors.fill: parent
            containmentMask: parent
            hoverEnabled: true
            acceptedButtons: Qt.LeftButton
            onClicked: {
                comp.checked = !comp.checked
                hovered = false     //等再次进入再点亮
                comp.checkedChanged()
            }
            onEntered:{
                hovered = true
            }
            onExited:{
                hovered = false
            }
        }
    }
}


