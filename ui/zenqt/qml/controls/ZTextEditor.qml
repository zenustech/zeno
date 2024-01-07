import QtQuick 2.12
import QtQuick.Controls 2.15

Rectangle{
    id: backGround
    width: 96
    height: 100

    color: "#191D21"
    border.color: control.enabled ? "#21be2b" : "transparent"

    property alias text: textInput.text

    MouseArea{
        anchors.fill: parent
        onWheel: (mouse)=> {
            mouse.accepted = true
        }
        Flickable {
            id: flick
            anchors.fill: parent
            contentWidth: textInput.contentWidth
            contentHeight: textInput.contentHeight
            clip: true

            function ensureVisible(r)
            {
                if (contentX >= r.x)
                    contentX = r.x;
                else if (contentX+width <= r.x+r.width)
                    contentX = r.x+r.width-width;
                if (contentY >= r.y)
                    contentY = r.y;
                else if (contentY+height <= r.y+r.height)
                    contentY = r.y+r.height-height;
            }
            TextEdit {
                id: textInput
                width: flick.width
                height: Math.max(flick.height, contentHeight)

                color: "#FFF"
                selectionColor: "#0078D7"
                font.pointSize: 12
                font.family: "Consolas"

                focus: false
                selectByMouse: true
                wrapMode: TextEdit.Wrap

                onCursorRectangleChanged: flick.ensureVisible(cursorRectangle)
                Keys.onEscapePressed: focus = false
            }
        }
    }
}