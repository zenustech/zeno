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

    property int xoffset: 4
    property real radius: 4
    property int side: 6
    property bool checked: false     //外部已定义alias
    property bool hovered: false
    property bool is_round_bottom: false    //底部是否有圆角，比如参数body显示的时候

    implicitWidth:  side + radius + xoffset
    implicitHeight: parent.height

    //signal checkedChanged

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

            startX: {
                return comp.xoffset
            }
            startY: 0
            PathLine { 
                x: comp.xoffset + comp.side
                y: 0
            }
            //右边button上方的弧
            PathArc {
                relativeX: comp.radius
                relativeY: comp.radius
                radiusX: comp.radius
                radiusY: comp.radius
                direction: PathArc.Clockwise
            }
            PathLine {
                x: comp.xoffset + comp.side + comp.radius
                y: comp.is_round_bottom ? (comp.height - comp.radius) : comp.height
            }
            PathArc {
                relativeX: comp.is_round_bottom ? -comp.radius : 0
                relativeY: comp.is_round_bottom ? comp.radius : 0
                radiusX: comp.is_round_bottom ? comp.radius : 0
                radiusY: comp.is_round_bottom ? comp.radius : 0
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


