import QtQuick                      2.7
import QtQuick.Controls             2.1
import QtQuick.Controls.Material    2.1
import QtQuick.Layouts              1.3

import QuickQanava 2.0 as Qan
//import "qrc:/QuickQanava"   as Qan

Qan.Navigable {
    id: navigable
    clip: true
    RowLayout {
        anchors.top: navigable.top
        anchors.horizontalCenter: navigable.horizontalCenter
        width: navigable.width / 2
        Slider {
            id: zoomSlider
            Layout.fillWidth: true
            to: navigable.zoomMax > 0. ? navigable.zoomMax : 10.
            from: navigable.zoomMin
            stepSize: 0.1
            onValueChanged: navigable.zoom = value
            property real rtValue: from + position * (to - from)
            onPositionChanged: navigable.zoom = rtValue
            Component.onCompleted: value = navigable.zoom
        }
        CheckBox {
            checked: navigable.zoomOrigin === Item.Center
            text: "Zoom on view center"
            onCheckedChanged: {
                navigable.zoomOrigin = checked ? Item.Center : Item.TopLeft
            }
        }
        CheckBox {
            checked: navigable.autoFitMode === Qan.Navigable.AutoFit
            text: "AutoFit"
            onCheckedChanged: navigable.autoFitMode = checked ? Qan.Navigable.AutoFit : Qan.Navigable.NoAutoFit
        }
        Button {
            text: "Fit in view"
            onClicked: navigable.fitInView()
        }
    } // RowLayout options
    Rectangle {
        anchors.right: navigable.right
        anchors.bottom: navigable.bottom
        width: 0.2 * parent.width
        height: 0.2 * parent.height
        opacity: 0.8
        border.width: 2
        border.color: Material.accent
        Qan.NavigablePreview {
            anchors.fill: parent; anchors.margins: 1
            source: navigable
            visible: true
        }
    }
    Image {
        parent: navigable.containerItem
        id: imageRenderer
        smooth: true
        antialiasing: true
        fillMode: Image.PreserveAspectFit
        source:  "qrc:/res/image.jpg"
    }
}
