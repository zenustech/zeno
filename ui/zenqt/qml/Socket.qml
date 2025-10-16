import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Shapes 1.6


Item {
    id: comp

    property bool input : true
    property var sockOnClicked
    property var mismatchSocket
    property var matchSocket
    property string paramName

    implicitWidth: shape.xline + shape.xradius
    implicitHeight: 2 * shape.yradius + shape.yline

    transform: Matrix4x4 {
        matrix: Qt.matrix4x4(input ? 1 : -1, 0, 0, input ? 0 :shape.xline + shape.xradius,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1)
    }

    Shape {
        id: shape
        anchors.fill: parent
        antialiasing: true

        property int xline: 6
        property int xradius: 5
        property int yradius: 7
        property int yline: 6
        property int ringwidth: shape.yradius * 0.7
        property int innerradius: 2
        property int innerYline: 2 * shape.yradius + shape.yline - 2 * shape.ringwidth - 2 * shape.innerradius
        property int innerXline: 1.0 * shape.xradius

        ShapePath {
            id: path
            strokeColor: "transparent"
            strokeWidth: 0
            fillColor: "#4b9ef4"

            startX: 0
            startY: 0
            PathLine { x: shape.xline; y: 0 }
            PathArc {
                relativeX: shape.xradius
                relativeY: shape.yradius
                radiusX: shape.xradius
                radiusY: shape.yradius
            }
            PathLine { relativeX: 0; relativeY: shape.yline}
            PathArc { 
                relativeX: -shape.xradius
                relativeY: shape.yradius
                radiusX: shape.xradius
                radiusY: shape.yradius
            }
            PathLine { relativeX: -shape.xline; relativeY: 0 }

            //start to draw inner path:
            PathLine { relativeX: 0; relativeY: -shape.ringwidth }
            PathLine { relativeX: shape.innerXline; relativeY: 0}
            PathArc {
                relativeX: shape.innerradius
                relativeY: -shape.innerradius
                radiusX: shape.innerradius
                radiusY: shape.innerradius
                direction: PathArc.Counterclockwise
            }
            PathLine { relativeX: 0; relativeY: -shape.innerYline }
            PathArc {
                relativeX: -shape.innerradius
                relativeY: -shape.innerradius
                radiusX: shape.innerradius
                radiusY: shape.innerradius
                direction: PathArc.Counterclockwise
            }
            PathLine { relativeX: -shape.innerXline; relativeY: 0}
            PathLine { x: 0; y: 0 }
        }

        //inner shape drawing
        ShapePath {
            id: innerpath
            strokeColor: "transparent"
            strokeWidth: 0
            fillColor: "#784B9EF4"

            startX: 0
            startY: shape.ringwidth

            PathLine {
                relativeX: shape.innerXline;
                relativeY: 0
            }
            PathArc {
                relativeX: shape.innerradius
                relativeY: shape.innerradius
                radiusX: shape.innerradius
                radiusY: shape.innerradius
            }
            PathLine {
                relativeX: 0
                relativeY: shape.innerYline
            }
            PathArc {
                relativeX: -shape.innerradius
                relativeY: shape.innerradius
                radiusX: shape.innerradius
                radiusY: shape.innerradius
            }
            PathLine {
                relativeX: -shape.innerXline
                relativeY: 0
            }
            PathLine {
                x: 0
                y: shape.ringwidth
            }
        }
    }

    MouseArea {
        id: socketMouseArea
        property double factor: 1.15
        layer.enabled: true
        layer.samples: 8
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
        drag.threshold: 0
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.horizontalCenterOffset: -parent.width
        anchors.verticalCenter: parent.verticalCenter
        width: parent.width * 4
        height: parent.height * 2


        onClicked: function() {
            //console.log("on clicked socket")
            comp.sockOnClicked(comp)
        }

        onEntered: function() {
            //console.log("onEnter socket......")
            comp.matchSocket(comp)
        }

        onExited: function() {
            //console.log("onExited......")
            comp.mismatchSocket()
        }
    }
}