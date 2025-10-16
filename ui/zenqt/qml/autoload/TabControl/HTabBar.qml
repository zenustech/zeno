// =========================================
// =============== 水平标签栏 ===============
// =========================================

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

RowLayout  {
    id: hTabBarLayout
    anchors.fill: parent
    spacing: 0

    // 标签栏控制（左）
    Rectangle  {
        visible: false // 先隐藏
        id: hTabBarCtrlLeft
        Layout.fillHeight: true
        width: parent.height
        color: "#00000000"
    }

    // 标签栏本体（中）
    Rectangle  {
        id: hTabBarMain
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: "#00000000"

        property int maxTabWidth: 200 // 单个标签最大宽度
        property int tabWidth: 200 // 标签当前宽度

        // 方法：重设标签按钮宽度
        function resetTabBtnWidth() {
            let w = hTabBarMain.width
            if(!app.tab.barIsLock) w -= tabBarControl.width // 无锁定时，减去+按钮宽度
            w = Math.round(w / tabBarRepeater.model.count)
            tabWidth = Math.min(w, maxTabWidth)
        }
        onWidthChanged: resetTabBtnWidth()  // 监听标签栏总宽度变化
        // 监听改变锁定，重设宽度
        property bool isLock: app.tab.barIsLock
        onIsLockChanged: {
            hTabBarMain.resetTabBtnWidth()
        }

        MouseArea { // 点击标签栏空余位置，都是添加新标签
            anchors.fill: parent
            onClicked: {
                if(!app.tab.barIsLock) tabBarRepeater.addNavi()
            }
        }

        Rectangle { // 标签按钮下方的阴影
            anchors.bottom: parent.bottom
            width: parent.width
            height: parent.height / 2
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#00000000" }
                GradientStop { position: 1.0; color: "#11000000" }
            }
        }

        Rectangle { // 拖拽时的位置指示器
            id: dragIndicator
            visible: false
            width: parent.tabWidth / 2
            height: parent.height
            gradient: Gradient { // 水平渐变
                orientation: Gradient.Horizontal
                GradientStop { position: 1.0; color: "#00000000" }
                GradientStop { position: 0.0; color: "#33000000" }
            }
        }

        // 水平标签栏行布局
        Row {
            id: hTabBarMainRow
            spacing: -2 // 给负的间隔，是为了让选中标签能覆盖左右两边标签的竖线

            // ===== 标签按钮组 =====
            TabBarRepeater {
                id: tabBarRepeater
                // 标签元素模板
                delegate: HTabButton {
                    title: title_ // 标题
                    index: index_ // 位序
                    checked: checked_ // 初始时是否选中
                    width: hTabBarMain.tabWidth
                    height: hTabBarMain.height
                }

                // 事件：按钮数量变化
                onCountChanged: hTabBarMain.resetTabBtnWidth()

                // ========================= 【拖拽相关】 =========================

                // 事件：创建新标签时（与父类的槽同时生效）
                onItemAdded: { 
                    // 链接表现相关的槽函数
                    item.dragStart.connect(dragStart)
                    item.dragFinish.connect(dragFinish)
                    item.dragMoving.connect(dragMoving)
                }

                property var intervalList: [] // 记录按钮位置区间的列表
                property var originalPosList: [] // 记录按钮初始位置的列表
                property int originalX // 记录本轮拖拽前，被拖拽按钮原本的位置
                function dragStart(index){ // 方法：开始拖拽
                    // 重新记录当前所有按钮的位置
                    originalX = itemAt(index).x
                    intervalList = [-Infinity] // 下限：负无穷
                    originalPosList = [itemAt(0).x]
                    for(let i=1, c=model.count; i < c; i++){ // 按钮位置区间
                        const it = itemAt(i)
                        intervalList.push(it.x)
                        originalPosList.push(it.x)
                    }
                    intervalList.push(Infinity) // 上限：负无穷
                    dragIndicator.visible = true

                }
                function btnDragIndex(index){ // 方法：返回当前index应该所处的序号
                    const dragItem = itemAt(index)
                    const x = dragItem.x + Math.round(dragItem.width/2) // 被拖动按钮的中心位置
                    let go = 0 // 应该拖放到的位置
                    for(const c=intervalList.length-1; go < c; go++){
                        if(x >= intervalList[go] && x <= intervalList[go+1]){
                            break;
                        }
                    }
                    return go;
                }
                function dragMoving(index, x){ // 方法：拖拽移动
                    let go = btnDragIndex(index) // 应该拖放到的序号
                    dragIndicator.x = originalPosList[go]
                }
                function dragFinish(index){ // 方法：结束拖拽
                    dragIndicator.visible = false
                    let go = btnDragIndex(index) // 应该拖放到的序号
                    if(index !== go){ // 需要移动
                        model.move(index, go, 1)
                        app.tab.move(index, go)
                    } else { // 无需移动，则回到原位
                        itemAt(index).x = originalX
                    }
                    resetTabIndex()
                }
            }
            
            // 元素：控制按钮
            Rectangle{
                id: tabBarControl
                color: "#00000000"
                property int size: 40
                Component.onCompleted: size = hTabBarMain.height
                width: size // w和h不能动态绑定到父组件height，可能会被意外置0
                height: size // 所以初始化时绑定一次即可。
                visible: !app.tab.barIsLock
                z: 100000
                x:100

                property color bgColorNormal: "#00000000" // 正常
                property color bgColorHovered: "#22000000" // 悬停
                property color bgColorPressed : "#44000000" // 按下
                // 添加“+”按钮
                Button {
                    anchors.centerIn: parent
                    implicitWidth: hTabBarMain.height - 10
                    implicitHeight: hTabBarMain.height - 10
                    font.pixelSize: hTabBarMain.height - 14
                    text: "+"
                    font.bold: true
                    background: Rectangle {
                        radius: 6
                        anchors.fill: parent
                        color: parent.pressed ? parent.parent.bgColorPressed: (
                            parent.hovered ? parent.parent.bgColorHovered : parent.parent.bgColorNormal
                        )
                    }
                    onClicked: {
                        tabBarRepeater.addNavi()
                    }
                }
            }

            // 动画
            add: Transition { // 添加子项
                NumberAnimation {
                    properties: "opacity, scale" // 透明度和大小从小到大
                    from: 0; to: 1.0
                    easing.type: Easing.OutBack // 缓动：超出反弹
                    duration: 400
                }
            }
            move: Transition { // 移动子项
                NumberAnimation {
                    properties: "x,y"
                    easing.type: Easing.OutBack
                    duration: 400
                }
            }
        }
    }

    // 标签栏控制（右）
    Rectangle  {
        id: hTabBarCtrlRight
        Layout.fillHeight: true
        width: parent.height
        color: "#00000000"
        // color: "blue"

        property color bgColorNormal: "#00000000" // 正常
        property color bgColorHovered: "#22000000" // 悬停
        property color bgColorChecked: "#66000000" // 选中
        // 锁定“🔒”按钮
        Button {
            anchors.centerIn: parent
            implicitWidth: hTabBarMain.height - 10
            implicitHeight: hTabBarMain.height - 10
            font.pixelSize: hTabBarMain.height - 14
            checkable: true
            checked: app.tab.barIsLock

            text: "L"
            font.bold: true
            background: Rectangle {
                radius: 6
                anchors.fill: parent
                color: parent.checked ? parent.parent.bgColorChecked: (
                    parent.hovered ? parent.parent.bgColorHovered : parent.parent.bgColorNormal
                )
            }
            onCheckedChanged: { // 双向绑定锁定标记
                app.tab.barIsLock = checked
            }
        }
    }
}