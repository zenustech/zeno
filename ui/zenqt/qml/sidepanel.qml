import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12

ApplicationWindow {
    id: window
    width: 360
    height: 520
    visible: true
    title: qsTr("Side Panel")

    readonly property bool inPortrait: window.width < window.height

    ToolBar {
        id: overlayHeader
        z: 1
        width: parent.width
        parent: window.overlay

        Label {
            id: label
            anchors.centerIn: parent
            text: "Qt Quick Controls 2"
        }
    }

    Drawer {
        id: drawer

        y: overlayHeader.height
        width: window.width / 2
        height: window.height - overlayHeader.height

        modal: inPortrait
        interactive: inPortrait
        position: inPortrait ? 0 : 1
        visible: !inPortrait

        ListView {
            id: listView
            anchors.fill: parent

            headerPositioning: ListView.OverlayHeader
            header: Pane {
                id: header
                z: 2
                width: parent.width
                contentHeight: logo.height
                Image {
                    id: logo
                    width: parent.width
                    source: "images/qt-logo.png"
                    fillMode: implicitWidth > width ? Image.PreserveAspectFit : Image.Pad
                }
                MenuSeparator {
                    parent: header
                    width: parent.width
                    anchors.verticalCenter: parent.bottom
                    visible: !listView.atYBeginning
                }
            }

            footer: ItemDelegate {
                id: footer
                text: qsTr("Footer")
                width: parent.width

                MenuSeparator {
                    parent: footer
                    width: parent.width
                    anchors.verticalCenter: parent.top
                }
            }

            model: 10

            delegate: ItemDelegate {
                text: qsTr("Title %1").arg(index + 1)
                width: listView.width
            }

            ScrollIndicator.vertical: ScrollIndicator { }

        }
    }

    Flickable {
        id: flickable

        anchors.fill: parent
        anchors.topMargin: overlayHeader.height
        anchors.leftMargin: !inPortrait ? drawer.width : undefined

        topMargin: 20
        bottomMargin: 20
        contentHeight: column.height

        Column {
            id: column
            spacing: 20
            anchors.margins: 20
            anchors.left: parent.left
            anchors.right: parent.right

            Label {
                font.pixelSize: 22
                width: parent.width
                elide: Label.ElideRight
                horizontalAlignment: Qt.AlignHCenter
                text: qsTr("Side Panel Example")
            }

            Label {
                width: parent.width
                wrapMode: Label.WordWrap
                text: qsTr("This example demonstrates how Drawer can be used as a non-closable persistent side panel.\n\n" +
                           "When the application is in portrait mode, the drawer is an interactive side panel that can " +
                           "be swiped open from the left edge. When the application is in landscape mode, the drawer " +
                           "and the content are laid out side by side.\n\nThe application is currently in %1 mode.").arg(inPortrait ? qsTr("portrait") : qsTr("landscape"))
            }
        }

        ScrollIndicator.vertical: ScrollIndicator { }
    }
}