// ================================================
// =============== 导航页（新标签页） ===============
// ================================================

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import ".."

TabPage {
    title: qsTr("新标签页")
    index: -1 // 导航页的排序为最小值，保证在所有页之前

    // =============== 逻辑 ===============

    id: naviPage
    
    ListModel { // 所有页面的标题
        id: pageModel
    }
    // 初始化数据
    onReady: ()=>{
        initData()
    }
    function initData() {
        pageModel.clear()
        const f = app.tab.fileList
        // 遍历所有文件信息（排除第一项自己）
        for(let i=1,c=f.length; i<c; i++){
            pageModel.append({
                "title": f[i].title,
                "intro": f[i].intro,
                "fileName": f[i].fileName,
                "fileIndex": i
            })
        }
    }
    // 动态变化的简介文本
    property string introText: qsTr(`# 欢迎使用 TabView-Demo
  
请选择切换一个页面。`)


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
                                        // 寻找本组件在pageList的下标
                                        const list = app.tab.pageList
                                        let i, c=list.length
                                        for(i=0; i<c; i++){
                                            if(list[i].obj===naviPage){
                                                break
                                            }
                                        }
                                        if(i === c) {
                                            console.log("Error: 未在pageList找到导航页对象！")
                                            return
                                        }
                                        app.tab.naviTabPage(i, fileIndex)
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