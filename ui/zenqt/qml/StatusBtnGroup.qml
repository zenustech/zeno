import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Shapes 1.6


Item {
    id: comp
    property int xoffset: 12
    property int side: 20
    property int fixheight: 48
    property real radius: 4
    property bool round_last_btn: true
    property bool isview: false
    property bool isbypass: false

    implicitWidth: 2 * comp.side + 2
    implicitHeight: fixheight

    StatusImgBtn{
        id: bypass_btn
        x: 0
        width: comp.xoffset + comp.side
        height: fixheight
        source: "qrc:/icons/bypass-off.svg"
        source_on: "qrc:/icons/bypass-on.svg"
        property bool hovered: false
        property alias checked: comp.isbypass

        onHoveredChanged: {
            if (bypass_btn.hovered) {
                statusImgGroup.visible = false;//true;
            }
        }

        onCheckedChanged: {
            statusImgGroup.visible = false;
        }
    }

    StatusImgBtn{
        id: view_btn
        x: comp.side + 3
        width: comp.side
        height: fixheight
        source: comp.round_last_btn ? "qrc:/icons/view-off.svg" : "qrc:/icons/view-noradius-off.svg"
        source_on: comp.round_last_btn ? "qrc:/icons/view-on.svg" : "qrc:/icons/view-noradius-on.svg"
        property bool hovered: false
        property alias checked: comp.isview

        onHoveredChanged: {
            if (view_btn.hovered) {
                statusImgGroup.visible = false;//true;
            }
        }

        onCheckedChanged: {
            statusImgGroup.visible = false;
        }
    }

    Item {
        id: statusImgGroup
        visible: false
        property real leftoffset: 20
        implicitWidth: imggroup_layout.implicitWidth + leftoffset
        implicitHeight: imggroup_layout.implicitHeight
        y: bypass_btn.y - imgByPass.height
        x: bypass_btn.x - leftoffset
        z: -100

        Rectangle {
            anchors.fill: parent
            //color: "red"
            z: -10
            color: "transparent"
        }

        ColumnLayout {
            id: imggroup_layout

            Rectangle{
                id: img_group
                width:childrenRect.width
                height:childrenRect.height
                color: "transparent"
                
                StatusImgBtn{
                    id: imgByPass
                    x: statusImgGroup.leftoffset + bypass_btn.x + comp.xoffset - 2
                    width: 64
                    height: 64
                    source: "qrc:/icons/MUTE_dark.svg"
                    source_on: "qrc:/icons/MUTE_light.svg"
                    property alias checked: comp.isbypass
                    property alias hovered: bypass_btn.hovered
                }
                StatusImgBtn{
                    id: imgView
                    width: 64
                    height: 64
                    x: imgByPass.x + imgByPass.width - imgByPass.xoffset - 2
                    source: "qrc:/icons/VIEW_dark.svg"
                    source_on: "qrc:/icons/VIEW_light.svg"
                    property alias checked: comp.isview
                    property alias hovered: view_btn.hovered
                }
            }
            Rectangle{
                id: rc_placeholder
                width: img_group.width
                height: 64
                color: "transparent"
            }

        }

        MouseArea {
            id: mouse_area_imggroup
            hoverEnabled: true
            anchors.fill: parent

            function isbypass(mousepos) {
                if (mousepos.x < imgView.x && mousepos.y < imgByPass.y + imgByPass.height) {
                    return true
                }
                else {
                    return false
                }
            }

            onPositionChanged: {
                if (isbypass(mouse)) {
                    imgByPass.hovered = true
                    imgView.hovered = false
                }
                else {
                    imgView.hovered = true
                    imgByPass.hovered = false
                }
                // console.log("mouse.x = " + mouse.x + " mouse.y = " + mouse.y)
                // console.log("parent.x = " + statusImgGroup.x)
                // console.log("imgByPass.x = " + imgByPass.x)
                // console.log("imgView.x = " + imgView.x)
            }

            onExited: {
                imgByPass.hovered = false
                imgView.hovered = false
                statusImgGroup.visible = false
            }

            onPressed: {
                if (isbypass(mouse)) {
                    imgByPass.checked = !imgByPass.checked
                }
                else{
                    imgView.checked = !imgView.checked
                }
                statusImgGroup.visible = false
            }
        }
    }
}