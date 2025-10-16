import QtQuick                   2.12
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3


Item {
    id: root

    property var model          //ParamGroupModel*
    implicitWidth: mainmain_layout.implicitWidth
    implicitHeight: mainmain_layout.implicitHeight

    //Body
    ColumnLayout {
        id: mainmain_layout
        anchors.fill: parent

        Repeater {
            id: repeater
            model: {
                return root.model
            }

            //显示一个group name
            delegate: ColumnLayout {
                Layout.fillWidth: true

                // Layout.preferredHeight: height
                // Layout.preferredWidth: width

                Button {
                    Layout.fillWidth: true
                    text: groupname
                    onClicked: propGroup.shown = !propGroup.shown
                    visible: root.model.rowCount() > 1   //只有一个group可能是默认的情况，不予以显示
                }

                //把整个group 显示出来
                ZPropPanel_Group {
                    id: propGroup
                    property bool shown: true
                    visible: shown
                    //height: implicitHeight
                    height: shown ? implicitHeight : 0
                    Behavior on height {
                        NumberAnimation {
                            easing.type: Easing.InOutQuad
                        }
                    }
                    //Behavior on height { NumberAnimation { duration: 200 } }

                    clip: true

                    Layout.fillWidth: true
                    model: params
                }
            }
        }
    }
}