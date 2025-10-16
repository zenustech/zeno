import QtQuick 2.12
import QtQuick.Layouts 1.3
import QtQuick.Controls.Styles 1.4
import zeno.enum 1.0


RowLayout {
    id: qmlparam
    property string arg_name
    property bool arg_isinput
    property int arg_control
    property var _controlObj : null
    property var _socketObj : null

    property var sockOnClicked
    property var mismatchSocket
    property var matchSocket
    spacing: 10

    /*
    Socket {}
    Text {
        text: qmlparam.arg_name
    }
    */

    function getSocketPos() {
        //console.log(qmlparam.parent.x + _socketObj.x)
        return {'x': _socketObj.x, 'y': _socketObj.y + _socketObj.height/2}
    }

    function getSocketItemObj() {
        return _socketObj
    }

    function createSocket(isInput) {
        var component = Qt.createComponent("qrc:/qml/Socket.qml");
        if (component.status == Component.Ready) {
            _socketObj = component.createObject(qmlparam)
            _socketObj.input = isInput
            _socketObj.sockOnClicked = qmlparam.sockOnClicked
            _socketObj.mismatchSocket = qmlparam.mismatchSocket
            _socketObj.matchSocket = qmlparam.matchSocket
            _socketObj.paramName = qmlparam.arg_name
        }
    }

    function createName() {
        //只有这种写法能binding c++类扩展的数据
        
        var item = Qt.createQmlObject('
            import QtQuick 2.12;
            SocketName {
                text: qmlparam.name
            }'
            ,qmlparam);

        /*
        var component = Qt.createComponent("qrc:/qml/SocketName.qml");
        if (component.status == Component.Ready) {
            var obj = component.createObject(qmlparam)
            obj.text = qmlparam.name
        }
        */
    }

    function createFillSpacer() {
        var component = Qt.createComponent("qrc:/qml/FillSpacer.qml");
        if (component.status == Component.Ready) {
            var obj = component.createObject(qmlparam)
        }
    }

    function createFixSpacer() {
        var item = Qt.createQmlObject('import QtQuick 2.12; Rectangle {color: "transparent"; width: 6; height: 1}',qmlparam);
    }

    Component.onCompleted: {
        if (qmlparam.arg_isinput) {
            createSocket(true)
            createName()
            createFillSpacer()
            createFixSpacer()
        }
        else {
            createFillSpacer()
            createName()
            createSocket(false)
        }
    }

    onArg_controlChanged: {
        if (_controlObj) {
            console.log("onArg_controlChanged")
            _controlObj.destroy();
        }
    }
}
