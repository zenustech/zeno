// ================================================
// =============== 导航页（新标签页） ===============
// ================================================

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import ".."

Rectangle {

    // =============== 逻辑 ===============

    id: naviPage
    anchors.fill: parent
    
    ListModel { // 所有页面的标题
        id: pageModel
    }
    
    // 初始化数据
    Component.onCompleted: initData()
    function initData() {
        pageModel.clear()
        const f = app.tab.infoList
        // 遍历所有文件信息（排除第一项自己）
        for(let i=1,c=f.length; i<c; i++){
            pageModel.append({
                "title": f[i].title,
                "intro": f[i].intro,
                "infoIndex": i,
            })
        }
    }
    // 动态变化的简介文本
    property string introText: qsTr(`# 欢迎使用
  
请选择切换一个功能页。`)


    // =============== 布局 ===============

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        Rectangle{
            Layout.alignment: Qt.AlignHCenter
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            Layout.maximumWidth: 1300
            color: "#00000000"

            RowLayout{
                anchors.fill: parent
                spacing: 20

                // =============== 左侧，展示所有标签页名称 ===============
                Rectangle {
                    width: 250
                    Layout.fillHeight: true
                    color: "#09000000"
                    radius: 20

                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: 20
                        clip: true

                        Column {
                            anchors.fill: parent
                            spacing: 10

                            Repeater {
                                model: pageModel
                                Button {
                                    text: title
                                    width: parent.width
                                    height: 50

                                    onHoveredChanged: {
                                        naviPage.introText = intro
                                    }
                                    onClicked: {
                                        let i = app.tab.getTabPageIndex(naviPage)
                                        if(i < 0){
                                            console.error("【Error】导航页"+text+"未找到下标！")
                                        }
                                        app.tab.changeTabPage(i, infoIndex)
                                    }
                                    
                                    background: Rectangle {
                                        radius: 10
                                        anchors.fill: parent
                                        property color bgColorNormal: "#11000000" // 正常
                                        property color bgColorHovered: "#44000000" // 悬停
                                        property color bgColorPressed : "#66000000" // 按下
                                        color: parent.pressed ? bgColorPressed: (
                                            parent.hovered ? bgColorHovered : bgColorNormal
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                // =============== 右侧，展示功能简介 ===============
                Rectangle {
                    id: introContainer
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#09000000"
                    radius: 20
                    property int margin: 30
                    
                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: parent.margin
                        clip: true

                         TextEdit {
                            width: introContainer.width - introContainer.margin*2
                            textFormat: TextEdit.MarkdownText // md格式
                            wrapMode: TextEdit.Wrap // 尽量在单词边界处换行
                            readOnly: true // 只读
                            selectByMouse: true // 允许鼠标选择文本
                            selectByKeyboard: true // 允许键盘选择文本
                            font.pointSize: 11
                            text: introText
                        }
                    }
                }
            }
        }
    }
}

// https://doc.qt.io/qt-5.15/qml-qtquick-textedit.html