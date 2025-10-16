import QtQuick 2.12
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.3


Item {
    id: root
    implicitWidth:  mainLayout.implicitWidth
    implicitHeight: mainLayout.implicitHeight
    property var value: ["0","0"]
    signal editingFinished

    function get_value() {
        var vec = []
        vec.push(xedit.text)
        vec.push(yedit.text)
        return vec
    }

    function isNumeric(str) {
        //console.log("str = " + str)
        return Number.isFinite(Number(str));
    }

    function formatWithSignificantDigits(num, significantDigits) {
        //int类型的数值在model已经转了，这里拿到的都是int，而无需在意小数点的事情
        if (isNumeric(num)) {
            return parseFloat(num.toPrecision(significantDigits)).toString();
        }
        else {
            return num  //公式或字符串
        }
    }

    RowLayout {
        id: mainLayout
        anchors.fill: parent
        spacing: 10
        VecEdit {
            id: xedit
            Layout.fillWidth: true
            text: formatWithSignificantDigits(value[0], 7)
            onEditingFinished: {
                root.editingFinished()
            }
        }
        VecEdit {
            id: yedit
            Layout.fillWidth: true
            text: formatWithSignificantDigits(value[1], 7)
            onEditingFinished: {
                root.editingFinished()
            }
        }
    }
}

