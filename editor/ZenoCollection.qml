import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs 1.2

Item {
    id: thisCollection

    property var descs: []
    property var currScene: null

    Component {
        id: compZenoScene

        ZenoScene {}
    }

    function addScene() {
        if (currScene != null)
            currScene.visible = false
        var scene = compZenoScene.createObject(thisCollection, {
            collection: thisCollection,
            focus: true,
            visible: true
        })
        currScene = scene
        return scene
    }

    ToolBar {
        z: 4
        Flow {
            anchors.fill: parent

            ToolButton {
                text: qsTr("Open")
                icon.name: 'document-open'
                onClicked: {
                    fileOpenDialog.open()
                }

                FileDialog {
                    id: fileOpenDialog
                    title: qsTr("Select a file to open")
                    folder: shortcuts.documents
                    nameFilters: [
                        qsTr("Zeno Graph files (*.zsg)"),
                    ]
                    onAccepted: {
                        print('opening file', fileOpenDialog.fileUrl)
                    }
                    onRejected: {
                        print('no file opened')
                    }
                }
            }

            ToolButton {
                text: qsTr("Apply")
                icon.name: 'edit-cut'
                onClicked: {
                    var r_scene = currScene.dumpScene()
                    var str_scene = JSON.stringify(r_scene)
                    rootWindow.appData.load_scene(str_scene)
                }
            }
        }
    }

    Component {
        id: compZenoAddNodeMenu

        Rectangle {
            id: thisMenu
            x: 300
            y: 300
            width: 100
            height: 100
            color: '#996'

            property var scene: null
            property var descs: []

            Component.onCompleted: {
                print('wq', x, y, parent)
            }
        }
    }

    function onAddNode(scene) {
        compZenoAddNodeMenu.createObject(scene.sceneRect, {
            scene: scene,
            descs: descs,
        })
    }

    Component.onCompleted: {
        var str_descs = rootWindow.appData.get_descriptors()
        thisCollection.descs = JSON.parse(str_descs)
        addScene()
    }
}
