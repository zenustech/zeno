import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0
import QtQuick.Controls.Styles   1.4
import Qt.labs.settings 1.1


TabButton {
	id: root
	property bool closable: true
	property color textClr: "white"
	property color textOnClr: "white"

	signal closeTab
	contentItem: RowLayout {
		id: row
		signal closeTab
		Item {
			width: 20
		}
		Text {
			id: txtLabel
			text: root.text
			font: root.font
			color: root.checked ? root.textClr : root.textOnClr
		}
		Item {
			width: 20
			visible: !root.closable
		}
		Item {
			width:  16
			height: 16
			visible: root.closable

			Image {
				id: closesvg
				visible: root.hovered
				sourceSize.width: parent.width
                sourceSize.height: parent.height
				smooth: true
                antialiasing: true
                source: "qrc:/icons/closebtn.svg"

				MouseArea {
                	anchors.fill: parent
                	hoverEnabled: true
                	onClicked: {
						root.closeTab()
                    }
                    onEntered: {
                    }
                    onExited: {
                    }
                }
			}
		}
	}
}