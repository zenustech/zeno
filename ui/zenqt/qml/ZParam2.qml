import QtQuick 2.12
import QtQuick.Layouts 1.3
import ZQuickParam 1.0
import QtQuick.Controls.Styles 1.4


ZQuickParam {
    id: thisdata
    RowLayout {
        Socket {
        }
        SocketName {
            text: thisdata.name
        }
    }
}