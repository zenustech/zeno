import QtQuick 2.2
import QtQuick.Window 2.2
import QtQuick.Controls 2.2
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
            selectable: true
            width: 400
            height: 120
            color: "red"
            MouseArea {
                anchors.fill: parent
                drag.target: parent
                onClicked: {
                    console.log('clicked');
                }
            }
        }
    }
}
