import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: qsTr("Zeno Editor")

    ZenoScene {
        id: currScene
        anchors.fill: parent
        focus: true
    }

    Button {
        text: qsTr("Apply")
        onClicked: {
            var r_scene = currScene.dumpScene()
            var str_scene = JSON.stringify(r_scene)
            applicationData.load_scene(str_scene)
        }
    }
}
