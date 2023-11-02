
import QtQuick              2.8
import QtQuick.Controls     2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts      1.3
import QtGraphicalEffects   1.0

import QuickQanava 2.0 as Qan

Qan.GroupItem {
    id: customGroup
    width: 400; height: 200

    // Note: This a completely custom group sample where you could customize group geometry,
    // appearance. If you just need a rectangular group, use RectGroupTemplate and default
    // QuickQanava styling options.

    // Customize your group appearance here
    // <---------------------  Begin Customization
    Rectangle {
        id: background
        z: 0
        anchors.fill: parent
        radius: 2; color: "yellow"
        border.color: Material.accent; border.width: 4
        clip: true

        // NOTE: background effects (glow and linear gradient) are not defined using layer.enabled and layer.effect
        // properties since it would enable mipmapping and caching on target item and produce lower quality results
        // at high scales. If you need high FPS and do not care of high level of zoom, you should consider using
        // effect layers (and pay the associed memory cost...).
    }
    // Custom group constants
    readonly property color groupColor: "blue"
    readonly property color backColor: "violet"
    LinearGradient {
      id: backgroundEffet
        anchors.fill: parent;
        anchors.margins: background.border.width / 2.
        z: 1
        source: background
        start: Qt.point(0.,0.)
        end: Qt.point(background.width, background.height)
        gradient: Gradient {
            id: backGrad
            GradientStop { position: 0.0; color: customGroup.groupColor }
            GradientStop {
                position: 1.0;
                color: Qt.tint( customGroup.groupColor, customGroup.backColor )
            }
        }
    } // LinearGradient
    Glow {
        source: background
        anchors.fill: parent
        color: Material.theme === Material.Light ? Qt.lighter( Material.foreground ) : Qt.darker( Material.foreground )
        radius: 12;     samples: 15
        spread: 0.25;   transparentBorder: true
    } // Glow
    Rectangle {
        id: content
        anchors.fill: parent; anchors.margins: background.border.width / 2.
        color: Qt.rgba(0,0,0,0) // === "transparent"
        z: 3
    } // Rectangle
    Label {
        z: 4
        text: group ? group.label : ""
        anchors.centerIn: parent
    }
    Label {
        z: 4
        text: "Custom group"
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom; anchors.bottomMargin: 15
    }
    Label {
        id: dndNotifier
        z: 4
        visible: false
        text: "Drop node"
        color: "red"
        font.bold: true
        anchors.centerIn: parent
    }
    // <--------------------- End Customization

    // QuickQanava interface (enable node DnD and selection), necessary for custom groups
    // not relying on Qan.RectGroupTemplate

    // Parent Qan.GroupItem (and qan::GroupItem) must be aware of the concrete group content item to
    // enable DnD for grouping/ungrouping nodes, see qan::GroupItem::container property documentation.
    // MANDATORY
    container: content   // NOTE: content.z _must_ be > to background.z, otherwise nodes drag and dropped into
                         // the group would be behing our background linear gradient effect (backgroundEffet)

    // Enable custom visual notifications when a node is dragged inside a groupe.
    // OPTIONAL

    // Emitted by qan::GroupItem when node dragging start
    onNodeDragEnter: { dndNotifier.visible = true }
    // Emitted by qan::GroupItem when node dragging ends
    onNodeDragLeave: { dndNotifier.visible = false}
}
