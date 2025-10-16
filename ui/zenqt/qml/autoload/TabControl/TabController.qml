// ==================================================
// =============== 标签整体的逻辑控制器 ===============
// ==================================================

/*

动态加载标签页的流程：
1. loadTabFiles 加载本地文件

*/

import QtQuick 2.15
import Qt.labs.folderlistmodel 2.15 // 读文件

Item {

    // ========================= 【常量】 =========================

    property string tabPagesPath: "../TabPages" // 标签页文件的路径

    // ========================= 【变量】 =========================

    /*  文件【加载状态】，即为 fileList 的加载状态。枚举定义在 Tabs.qml 。
        Null：未加载 | Loading：加载中 | Ready：已就绪 | Error：出现异常 */
    property var fileStatus: Tabs.Status.Null
    property string errorString: "" // 加载失败的信息

    /*  【文件列表】：所有标签页的原始文件信息。
        每一项为字典： title: 组件标题    fileName:文件名    com:【组件类】   intro: 简介
        [0] 为默认加载的页（导航页），其余项的排列顺序就是它们在导航页中的排列顺序。*/
    property var fileList: []

    /*  【页面列表】：当前已加载到标签栏的页面对象信息。
        每一项为字典： title: 组件名称    fileName: 类的文件名     obj:【组件对象】 */
    property var pageList: []

    // 标签栏按钮对应的fileName，用以记忆打开软件的默认标签。顺序与pageList一致
    property var barFileNameList: [] 
    property int barCheckedIndex: -1 // 记录当前选中项的下标。
    property bool barIsLock: false // 当前标签栏布局是否已锁定。
    property var tabBarRepeater: null // 标签栏控制器的引用
    
    // ========================= 【执行：初始化】 =========================

    // 事件：启动时加载文件
    Component.onCompleted: {
        // 推迟到下一循环，等静态元素初始化完毕后才开始加载文件
        Qt.callLater(loadTabFiles) 
    }

    // 函数：从列表list中寻找下标为a或[keyName]为a的项。
    function findList(list, key, keyName){
        if(typeof key === "number"){
            if(key >= 0 && key < list.length)
                return list[key]
        }
        else if(typeof key === "string"){
            for(let i in list){
                if(list[i][keyName] === key)
                    return list[i]
            }
        }
        return null
    }

    // 方法：从文件列表中查找项。key可传入fileList的下标，或fileName。
    function getFile(key){ return findList(fileList, key, "fileName") }
    
    // 方法：从页面列表中查找项。key可传入pageList的下标，或fileName。
    function getPage(key){ return findList(pageList, key, "fileName") }

    // 初始流程1：启动加载标签页文件（异步）。
    property var objCache // 缓存字典，fileName为key，放置读文件初始化时创建的组件对象，后续可以拿出来使用
    function loadTabFiles() {
        objCache = {}
        fileStatus = Tabs.Status.Loading // 标记：正在加载

        // 创建文件加载组件，传入回调函数。
        const loadFileObj = loadTabFilesCom.createObject(null, {onLoadFunc: onLoadTabFiles});

        // 事件：文件组件加载完成。传入 model 为 FolderListModel 组件的引用。
        function onLoadTabFiles(model) {

            // 函数：从路径path加载qml文件并解析成fileList格式，返回信息字典。
            function getComInfo(path, fileName="") {
                const com = Qt.createComponent(path) // 尝试加载文件
                if (com.status === Component.Error) {  // 加载失败，提取错误信息
                    const str = com.errorString()
                    const last = str.lastIndexOf(":")
                    if(last < 0) last = -1
                    console.log(`加载失败：【${fileName}】${str.substring(last+1).replace("\n","")}`)
                    return null
                }
                else if (com.status !== Component.Ready){
                    console.log(`加载异常：【${fileName}】`)
                    return null
                }
                else {
                    // 实例化组件，先挂到缓存父级下
                    const obj = com.createObject(pageCache)
                    const dic = {
                        title: obj.title, // 标题
                        index: obj.index, // 参考排序
                        intro: obj.intro, // 简介
                        fileName: fileName, // 文件名
                        com: com // 源类，可用createObject实例化
                    }
                    objCache[fileName] = obj
                    return dic
                }
            }

            // 遍历读取到的tabPages文件夹，将合法信息填入fileList。
            let fs = []
            for(let i=0, c=model.count; i<c; i++){ 
                const fileName = model.get(i,"fileName") // 文件夹名
                const path = tabPagesPath+"/"+fileName+"/"+fileName+".qml" // 拼接page入口文件路径
                const dic = getComInfo(path, fileName)
                if(dic !== null){
                    fs.push(dic)
                }
            }
            if(fs.length <= 0){
                fileStatus = Tabs.Status.Error // 加载失败
                errorString = qsTr("未在[ %1 ]目录下找到有效的标签页文件。").arg(tabPagesPath)
                return
            }
            // 按照index，从小到大排序。
            fs.sort((a,b) => a.index-b.index)  
            // 检查index重复
            for(let i=0,c=fs.length; i<c; i++){ 
                if(i<c-1 && fs[i].index===fs[i+1].index){
                    // TODO: 发出警告
                    console.log(`【Warning】index重复：${fs[i].fileName}与${fs[i+1].fileName}均为${fs[i].index}！`)
                }
                delete fs[i].index // 删除该属性
            }
            fileList = fs // 记入全局变量
            loadFileObj.destroy() // 删除文件监听组件，防止后续文件变化引发重新加载
            initBarData() // 调用下一步：初始化标签栏数据
        }
    }

    // 初始流程2：初始化标签栏的数据部分。
    function initBarData() {
        let bf = [...barFileNameList] // 深拷贝一份来加工
        // 遍历标签栏记忆的列表，排除其中现在不存在的组件
        for(let i=bf.length-1; i>=0; i--){
            if(!getFile(bf[i]))
                bf.splice(i, 1)
        }
        // 若列表已空，则添加默认标签页
        if(bf.length===0){ 
            const naviCom = getFile(0) // 获取默认标签页
            bf.push(naviCom.fileName) // 向标签栏添加这个页文件名
            barCheckedIndex = 0 // 刷新选中下标
        }
        // 否则，排除重复项（TODO）
        else {
            let temp = []
            for(let i in bf){
                if(!temp.includes(bf[i]))
                    temp.push(bf[i])
            }
            bf = temp
        }
        // 检查选中下标的合法性
        if(barCheckedIndex<0) barCheckedIndex = 0
        else if(barCheckedIndex>bf.length-1)
            barCheckedIndex = bf.length-1
        barFileNameList = bf // 计入全局
        initPageObj() // 调用下一步：初始化页面对象
    }

    // 初始流程3：初始化标签页的对象。使用objCache中已缓存的对象。
    function initPageObj() {
        for(let i in barFileNameList){
            const obj = objCache[barFileNameList[i]] // 从缓存中获取对象
            delete objCache[barFileNameList[i]] // 删除缓存的引用
            const com = getFile(barFileNameList[i]) // 获取组件信息
            pageList.push({ // 填入对象列表
                title: com.title,
                obj: obj,
                fileName: com.fileName
            })
            obj.parent = pageContainer // 父级挂载到展示区
        }
        // 销毁未被取用的组件对象
        for(let i in objCache){
            objCache[i].destroy()
        }
        objCache = {} // 清空缓存
        
        fileStatus = Tabs.Status.Ready // 加载完毕
        // 调用各已加载页的启动方法
        Qt.callLater(// 在下一个时间循环执行
            function(){ 
                for(let i in pageList){
                    const obj = pageList[i].obj
                    if(typeof obj.onReady === "function")
                        obj.onReady()
                    else
                        console.log(`Error: 调用第${i}个-${obj.title}启动方法失败，类型为${obj.onReady}`)
                }
            }
        )
    }
    
    // ========================= 【执行：动态】 =========================

    // 函数：返回 index 是否是 pageList 的有效下标
    function isPageIndex(index){ return (index>=0 && index<pageList.length) }

    // 方法：添加一个页面。成功后调用 tabBarRepeater.onAddTab
    // index: 插入位置，  fileIndex: 在 fileList 中的位置
    function addTabPage(index, fileIndex){
        if(index<0) index=pageList.length // 表示在尾部添加
        else if(index>pageList.length){
            console.log("Error: 在"+index+"添加新页超出范围！")
            return
        }
        const file = getFile(fileIndex)
        if(file == null){
            console.log("Error: 添加文件序号为"+fileIndex+"的新页无效！")
            return
        }
        // 实例化组件，挂到展示区
        const obj = file.com.createObject(pageContainer)
        const page = {
            title: obj.title, // 标题
            fileName: file.fileName, // 文件名
            obj: obj // 页面对象
        }
        pageList.splice(index, 0, page)  // 列表添加
        barFileNameList.splice(index, 0, file.fileName)
        // 回调
        if (typeof tabBarRepeater.onAddTab === "function") {
            tabBarRepeater.onAddTab(index, fileIndex)
        }
        if (typeof obj.onReady === "function") {
            Qt.callLater(obj.onReady)  // 下一循环调用启动方法
        }
        settings.save() // 刷新保存
    }

    // 方法：页面导航，即 将位置为index的导航页替换成fileIndex页。
    function naviTabPage(index, fileIndex){
        if(!isPageIndex(index)){
            console.log("Error: "+index+"页导航无效！")
            return
        }
        const file = getFile(fileIndex)
        if(file == null){
            console.log("Error: 导航文件序号为"+fileIndex+"的新页无效！")
            return
        }
        // 实例化组件，挂到展示区
        const obj = file.com.createObject(pageContainer)
        pageList[index].obj.destroy()  // 删除旧的页对象
        // 添加新的页对象
        pageList[index] = { // 刷新列表
            title: obj.title, // 标题
            fileName: file.fileName, // 文件名
            obj: obj // 页面对象
        }
        barFileNameList[index] = file.fileName // 重新设置名称列表
        // 回调
        if (typeof tabBarRepeater.onNaviTab === "function") {
            tabBarRepeater.onNaviTab(index, fileIndex)
        }
        if (typeof obj.onReady === "function") {
            Qt.callLater(obj.onReady)  // 下一循环调用启动方法
        }
        settings.save() // 刷新保存
        // 若当前页是选中状态，则重新设置选中。
        if(barCheckedIndex == index){ 
            selectTabPage(index)
        }
    }

    // 方法：删除index页。成功后调用 tabBarRepeater.onDelTab
    function delTabPage(index){
        if(!isPageIndex(index)){
            console.log("Error: 删除"+index+"页无效！")
            return
        }
        pageList[index].obj.destroy()  // 页对象删除
        pageList.splice(index, 1)  // 列表删除
        barFileNameList.splice(index, 1)
        // 回调
        if (typeof tabBarRepeater.onDelTab === "function") {
            tabBarRepeater.onDelTab(index)
        }
        settings.save() // 刷新保存
    }

    // 方法：设置index页为选中。
    function selectTabPage(index){
        if(!isPageIndex(index)){
            console.log("Error: 选定"+index+"页无效！")
            return
        }
        barCheckedIndex = index
        // 遍历，将选中的页面设为激活状态，其他页面设为非激活状态
        for(let i in pageList){
            pageList[i].obj.isActive = (i==index)
        }
    }

    // 方法：将一个原本在index的页移到go处
    function move(index, go){
        var x = pageList.splice(index, 1)[0] // 删除
        pageList.splice(go, 0, x) // 添加
        var x = barFileNameList.splice(index, 1)[0]
        barFileNameList.splice(go, 0, x)
    }

    // ========================= 【辅助组件】 =========================

    // 缓存区，作为刚创建的页组件对象的父级。非可视
    Item { 
        id: pageCache
        visible: false
    }

    // 展示区，作为激活的页组件对象的父级。可挂载到可视节点下来展示。
    Item {
        id: pageContainer
        anchors.fill: parent

        // Rectangle{
        //     anchors.fill: parent
        //     // color: "blue"
        //     Text {
        //         anchors.centerIn: parent
        //         text: "展示区！！！"
        //     }
        // }
    }
    property var pageContainer: pageContainer

    // 动态加载的组件原型，用于辅助读取文件
    Component {
        id: loadTabFilesCom
        FolderListModel {
            folder: Qt.resolvedUrl(tabPagesPath) // 标签页文件的路径
            showDirs: true // 显示目录
            showFiles: false // 不显示文件
            property var onLoadFunc: undefined // // 加载完成后的回调函数
            onStatusChanged:{ // 监听文件夹加载完成
                if (status === FolderListModel.Ready){
                    if (typeof onLoadFunc === "function") {
                        onLoadFunc(this)
                    }
                }
            }
        }
    }

    visible: false
}