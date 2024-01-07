import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Shapes 1.6


Item {
    id: comp
    property int xoffset: 22
    property int side: 35

    implicitWidth: 3 * comp.side
    implicitHeight: parent.height

    MouseArea{
        id: mouseArea
        anchors.fill: parent
        hoverEnabled: true
        onExited: {
            if(!groupContainsMouse())
                statusImgGroup.visible = false
        }

        StatusBtn {
            id: once_btn
            basefillcolor: "#FFBD21"
            height: parent.height
            xoffset: comp.xoffset
            side: comp.side
            onStatusChanged: (status)=> {
                statusImgGroup.visible = true
                imgOnce.source = status ? "qrc:/icons/ONCE_light.svg" : "qrc:/icons/ONCE_dark.svg"
            }
        }

        StatusBtnSeperator {
            xoffset: comp.xoffset
            x: comp.side
        }

        StatusBtn {
            id: mute_btn
            basefillcolor: "#E302F8"
            height: parent.height
            xoffset: comp.xoffset
            side: comp.side
            x: comp.side + 1
            onStatusChanged: (status)=> {
                statusImgGroup.visible = true
                imgMute.source = status ? "qrc:/icons/MUTE_light.svg" : "qrc:/icons/MUTE_dark.svg"
            }
        }

        StatusBtnSeperator {
            xoffset: comp.xoffset
            x: 2 * comp.side + 1
        }

        StatusBtn {
            id: view_btn
            basefillcolor: "#30BDD4"
            height: parent.height
            xoffset: comp.xoffset
            side: comp.side
            lastBtn: true
            x: comp.side * 2 + 2
            onStatusChanged: (status)=> {
                statusImgGroup.visible = true
                imgView.source = status ? "qrc:/icons/VIEW_light.svg" : "qrc:/icons/VIEW_dark.svg"
            }
        }
    }
    Rectangle{
        id: statusImgGroup
        anchors.bottom: mouseArea.top

        width:childrenRect.width
        height:childrenRect.height
        color: "transparent"
        visible: false

        StatusImgBtn{
            id: imgOnce
            x: once_btn.x + comp.xoffset - 4
            source: "qrc:/icons/ONCE_dark.svg"
            onClickedSig: once_btn.mouseAreaAlias.doClick()
            onEnteredSig: once_btn.mouseAreaAlias.entered()
            onExitedSig: {
                once_btn.mouseAreaAlias.exited()
                if(!groupContainsMouse())
                    statusImgGroup.visible = false
            }
        }
        StatusImgBtn{
            id: imgMute
            x: mute_btn.x + comp.xoffset - 2
            source: "qrc:/icons/MUTE_dark.svg"
            onClickedSig: mute_btn.mouseAreaAlias.doClick()
            onEnteredSig: mute_btn.mouseAreaAlias.entered()
            onExitedSig: {
                mute_btn.mouseAreaAlias.exited()
                if(!groupContainsMouse())
                    statusImgGroup.visible = false
            }
        }
        StatusImgBtn{
            id: imgView
            x: view_btn.x + comp.xoffset
            source: "qrc:/icons/VIEW_dark.svg"
            onClickedSig: view_btn.mouseAreaAlias.doClick()
            onEnteredSig: view_btn.mouseAreaAlias.entered()
            onExitedSig: {
                view_btn.mouseAreaAlias.exited()
                if(!groupContainsMouse())
                    statusImgGroup.visible = false
            }
        }
    }
    function groupContainsMouse(){
        return once_btn.mouseAreaAlias.containsMouse || mute_btn.mouseAreaAlias.containsMouse || view_btn.mouseAreaAlias.containsMouse ||
        imgOnce.mouseArea.containsMouse || imgMute.containsMouse || imgView.containsMouse
    }
}