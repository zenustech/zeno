import QtQuick                   2.12
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3


Item {
    id: root
    property color bgcolor: "blue"
    property int radius
    property bool top_radius: true

    Rectangle {
        id: upper
        anchors.fill: parent
        color: root.bgcolor
        radius: root.radius
    }

    Rectangle {
        id: bottomer
        x: 0
        y: root.top_radius ? root.radius : 0
        width: parent.width
        height: parent.height - root.radius

        color: root.bgcolor
    }
}