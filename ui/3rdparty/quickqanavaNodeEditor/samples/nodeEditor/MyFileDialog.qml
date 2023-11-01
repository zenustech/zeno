import QtQuick 2.3
import QtQuick.Dialogs 1.2
import QtQuick.Controls 2.1

Item{
    property alias backGroundColor:labels.backGroundColor

    MySingleText{
        id: labels
        txt: qsTr(",,,")
        width: parent.width*0.75
        height: parent.height
        anchors.left:parent.left
    }
    Button {
        id:openBtn
        width: parent.width*0.25
        height: parent.height
        text:qsTr("view...")
        anchors.left: labels.right
        //anchors.leftMargin: 10
        onClicked: {
            fds.open();
        }


        FileDialog {
            id:fds
            title: "选择文件"
            folder: shortcuts.desktop
            selectExisting: true
            selectFolder: true
            selectMultiple: false
            //nameFilters: ["json文件 (*.json)"]
            onAccepted: {
                labels.txt = fds.fileUrl;
                console.log("You chose: " + fds.fileUrl);
            }

            onRejected: {
                labels.txt = "";
                console.log("Canceled");
                //Qt.quit();
            }

        }
    }
}