import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import ZQuickParam 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4


ApplicationWindow {
    id: appWindow
    visible: true
    title: qsTr("Hello World")

    ColumnLayout {

        spacing: 0

        Rectangle {
            //default property alias data: headerLayout.data
            implicitWidth: headerLayout.implicitWidth
            implicitHeight: headerLayout.implicitHeight
            Layout.fillWidth: true

            color: "red"
            id: headerItem

            RowLayout {
                id: headerLayout
                spacing: 50
                anchors.fill: parent

                Text {
                    id: centerText;
                    text: "A Single Text.";
                    font.pixelSize: 12;
                    font.bold: true;
                }
                Button {
                    id: button1
                    text: centerText.text
                    Layout.fillWidth: true
                }
            }
        }

        Rectangle {
            implicitWidth: bodyLayout.implicitWidth
            implicitHeight: bodyLayout.implicitHeight

            color: "green"
            id: bodyItem
            Layout.fillWidth: true

            RowLayout {
                id: bodyLayout
                spacing: 20
                anchors.fill: parent

                Text {
                    id: bodyText
                    text: "Bodyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"
                    font.pixelSize: 12;
                    font.bold: true;
                }
                Button {
                    id: button2
                    text: "body"
                    Layout.fillWidth: true
                }
            }
        }

        Rectangle {
            implicitWidth: footerLayout.implicitWidth
            implicitHeight: footerLayout.implicitHeight
            color: "yellow"
            Layout.fillWidth: true

            RowLayout {
                id: footerLayout
                spacing: 20
                anchors.fill: parent

                Text {
                    text: "src"
                    font.pixelSize: 12;
                    font.bold: true;
                }

                Rectangle {
                    id: button3
                    height: 50
                    Layout.fillWidth: true
                    color: "black"
                }
            }
        }
    }

}