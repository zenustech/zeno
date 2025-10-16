// =====================================================
// =============== 标签栏的复制器的逻辑处理 ===============
// =====================================================

// 页面表现（布局、动画）应该在继承的子类中编写

import QtQuick 2.15
import QtQuick.Controls 2.15

Repeater {
    // ========================= 【属性与变量】 =========================

    // delegate: // 子类中填入按钮组件，属性必须包含：
    //       title: title_     // 标题
    //       index: index_ // 位序
    //       checked: checked_ // 初始时是否选中
    //       width: 宽高
    //       height: 宽高
    model: ListModel{} // 标签元素列表，初始为空
    property var fileStatus: app.tab.fileStatus // 绑定到状态
    property var barNameList // 名称列表，与 app.tab.barFileNameList 对应

    // ========================= 【执行】 =========================

    // 初始流程3之后：加载初始标签
    onFileStatusChanged: {
        if(fileStatus === Tabs.Status.Ready){
            initBarTitles()
        }
    }
    function initBarTitles(){
        app.tab.tabBarRepeater = this // 保存引用
        barNameList = []
        for(let i=0,c=app.tab.pageList.length; i<c; i++){
            const title = app.tab.getPage(i).title
            model.insert(i, {
                "title_": title,
                "index_": i,
                "checked_": i === app.tab.barCheckedIndex // 初始选中
            })
        }
    }

    // 方法：在尾部添加一个导航页，并跳转到它
    function addNavi() {
        app.tab.addTabPage(-1, 0)
    }
    
    // 事件：收到在index处添加 序号为fileIndex的信息 的标签
    function onAddTab(index, fileIndex) {
        const info = app.tab.getFile(fileIndex)
        const title = info.title
        model.insert(index, {
            "title_": title,
            "index_": index,
            "checked_": true, // 选中新页
        })
    }

    // 方法：移除下标为index的标签页
    function delTab(index) { app.tab.delTabPage(index) }

    // 事件：收到移除下标为index标签页的指令
    function onDelTab(index) {
        if(itemAt(index).checked && model.count > 1) { // 跳转到最近的标签页
            // 如果当前不是末尾，则跳到下一页，且下一页下标-1。否则直接跳到上一页。
            if(index+1 < model.count) {
                const item = itemAt(index+1)
                item.index-- // 必须先刷新下标（不等resetTabIndex自动刷新），再选中。
                item.checked = true
            }else{
                itemAt(index-1).checked = true
            }
        }
        model.remove(index) // 删除按钮
        let s = ""
        for(let i in app.tab.pageList){
            s += "|"+i+" "+app.tab.pageList[i].title
            if(i == this)
                s+="√"
        }
    }

    // 事件：收到页面导航的指令
    function onNaviTab(index, fileIndex) {
        const info = app.tab.getFile(fileIndex)
        const title = info.title
        model.set(index, {"title_": title})
    }

    // 方法：改变选中页
    // function selectTab(index) { app.tab.selectTabPage(index) }
    function selectTab(index) {
        app.tab.selectTabPage(index)
    }

    // 方法：重设所有标签组件的序号，防止增删后别的标签的位置错乱
    function resetTabIndex() { 
        for(let i=0, c=model.count; i < c; i++){
            itemAt(i).index = i
        }
    }

    // ========================= 【信号槽相关】 =========================

    onItemAdded: { // 创建新标签时：
        resetTabIndex() // 重设序号
        // 连接逻辑相关的槽函数
        item.toDel.connect(delTab) // 删除
        item.toChecked.connect(selectTab) // 选中
    }
    onItemRemoved: { // 移除标签时：
        if(model.count <= 0){
            // 若为空，则补充一个新标签
            Qt.callLater(addNavi) // 在下一个时间循环执行
        } else {
            resetTabIndex() // 否则，重设序号
        }
    }

}