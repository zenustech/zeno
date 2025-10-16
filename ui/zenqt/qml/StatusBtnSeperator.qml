import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Shapes 1.6

Item {
    property int xoffset: 22

    Shape {
        anchors.fill: parent
        antialiasing: true

        ShapePath {
            id: path
            strokeColor: "#484848"
            strokeWidth: 1
            fillColor: "transparent"
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
            PathLine { x: 0; y: comp.height }
        }
    }

}