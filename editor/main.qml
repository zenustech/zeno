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

        function doSelect(item) {
            console.log('doSelect', item)
        }

        Rectangle {
            width: 400
            height: 120
            radius: 5.0
            color: "red"
            MouseArea {
                anchors.fill: parent
                drag.target: parent
                onClicked: {
                    scene.doSelect(parent)
                }
            }
        }
    }
}
