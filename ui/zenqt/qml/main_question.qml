import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import SelfDefinedQMLType 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4


ApplicationWindow {
    id: appWindow
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Rectangle {
        id: rect1
        x: 0
        y: 0
        visible: true
        anchors.fill: parent
        color: "steelblue"
        Keys.enabled: true
        focus: true

        SelfDefinedQMLType {
            id: selfDefined
        }
        

        Keys.onPressed: {
            switch (event.key)
            {
            case Qt.Key_Left:
                console.log("Qt.Key_Left was pressed!!!")
                //don't know why,the following repor error, during execution.
                selfDefined.changeColor()       //Error:TypeError: Cannot call method 'changeColor' of null
                break;
            }
        }
    }

}