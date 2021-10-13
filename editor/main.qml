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

        property var selectedChildren: []

        function doSelect(item, multiselect) {
            if (item == null) {
                for (var i in selectedChildren)
                    selectedChildren[i].selected = false
                selectedChildren = []
            } else if (multiselect) {
                if (!selectedChildren.includes(item)) {
                    selectedChildren.push(item)
                    item.selected = true
                } else {
                    selectedChildren.remove(item)
                    item.selected = false
                }
            } else {
                if (selectedChildren.length == 1 && selectedChildren.includes(item)) {
                    for (var i in selectedChildren)
                        selectedChildren[i].selected = false
                    selectedChildren = []
                    item.selected = false
                } else {
                    for (var i in selectedChildren)
                        selectedChildren[i].selected = false
                    selectedChildren = [item]
                    item.selected = true
                }
            }
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                scene.doSelect(null)
            }
        }

        Rectangle {
            width: 400
            height: 120
            radius: 5.0
            color: '#aaa'
            border.color: '#6cf'
            border.width: selected ? 4 : 0

            property var selected: false

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
