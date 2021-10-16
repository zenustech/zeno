import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.2

ApplicationWindow {
    id: applicationWindow
    visible: true
    width: 640
    height: 480
    title: qsTr("Zeno Editor")

    ZenoScene {
        id: currScene
        anchors.fill: parent
        focus: true
    }

    header: ToolBar {
        Flow {
            anchors.fill: parent

            ToolButton {
                text: qsTr("Open")
                icon.name: 'document-open'
                onClicked: {
                    fileOpenDialog.open()
                }

                FileDialog {
                    property var choice: null
                    id: fileOpenDialog
                    title: qsTr("Select an image file")
                    folder: shortcuts.documents
                    nameFilters: [
                        qsTr("Image files (*.png *.jpeg *.jpg)"),
                    ]
                    onAccepted: {
                        fileOpenDialog.choice = fileOpenDialog.fileUrl
                    }
                    onRejected: {
                        fileOpenDialog.choice = null
                    }
                }
            }

            ToolButton {
                text: qsTr("Apply")
                icon.name: 'edit-cut'
                onClicked: {
                    var r_scene = currScene.dumpScene()
                    var str_scene = JSON.stringify(r_scene)
                    applicationData.load_scene(str_scene)
                }
            }
        }
    }
}
