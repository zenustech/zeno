import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.12

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: qsTr("Zeno Editor")

    Flickable {
        id: scene
        anchors.fill: parent
        boundsBehavior: Flickable.StopAtBounds
        clip: true
        interactive: true

        Rectangle {
            anchors.fill: scene
            width: 400
            height: 120
            color: "red"
            MouseArea {
                anchors.fill: parent
                drag.target: parent
                hoverEnabled: true
                onPositionChanged: {
                    hinter.x = mouseX - hinter.width / 2
                }
            }
            Rectangle {
                id: hinter
                width: 20
                height: 40
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "lightsteelblue" }
                    GradientStop { position: 1.0; color: "blue" }
                }
            }
        }
    }
}
