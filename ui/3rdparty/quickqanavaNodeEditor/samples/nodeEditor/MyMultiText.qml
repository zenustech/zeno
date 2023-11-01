import QtQuick                   2.3
import QtQuick.Controls          2.2
Item{
    property alias txt:contentText.text
    property alias backGroundColor:bkg.color
    property var textColor: "#FFFFFF"
    
    ScrollView {
        id: scView
        //anchors.centerIn: parent
        anchors.fill: parent
        background: Rectangle {
            id:bkg
            anchors.fill: parent
            border.color: "gray"
            radius: 5
        }
        ScrollBar.vertical.policy: ScrollBar.AlwaysOff
        //verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        TextArea {
            id: contentText
            property int preContentHeight: 0
            color: textColor
            wrapMode: TextArea.Wrap; selectByMouse: true;
            onContentHeightChanged: {
                //每一行为高度为14， 当输入大于3行的时候自动滚动
                if(contentHeight > 14 && contentHeight < 56) {
                    if(contentHeight != preContentHeight) {
                        preContentHeight = contentHeight;
                        scView.height += 14;
                    }
                }
            }
        }
    }
}