import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.12

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: qsTr("Zeno Editor")
    color: '#222'
    Flickable {
        id: zenoScene;

        Component {
            id: compZenoNode;
            ZenoNode {}
        }

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
                zenoScene.doSelect(null)
            }
        }

        Component.onCompleted: {
            compZenoNode.createObject(this);
        }
    }
}
