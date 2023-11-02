import QtQuick          2.12
import QtQuick.Controls 2.0
import QtQuick.Layouts  1.3
import QtQuick.Shapes   1.0

import QuickQanava          2.0 as Qan
import "qrc:/QuickQanava"   as Qan

Item {
    Qan.LineGrid { id: lineGrid }

    Qan.Navigable {
        id: navigable
        anchors.fill: parent
        clip: true
        navigable: true
        grid: lineGrid
        PinchHandler {
            target: null
            onActiveScaleChanged: {
                console.error('centroid.position=' + centroid.position)
                console.error('activeScale=' + activeScale)
                var p = centroid.position
                var f = activeScale > 1.0 ? 1. : -1.
                navigable.zoomOn(p, navigable.zoom + (f * 0.03))
            }
        }

        Rectangle {
            parent: navigable.containerItem
            x: 100; y: 100
            width: 50; height: 25
            color: "lightblue"
        }
        Rectangle {
            parent: navigable.containerItem
            x: 300; y: 100
            width: 50; height: 25
            color: "red"
        }
        Rectangle {
            parent: navigable.containerItem
            x: 300; y: 300
            width: 50; height: 25
            color: "green"
        }
        Rectangle {
            parent: navigable.containerItem
            x: 100; y: 300
            width: 50; height: 25
            color: "blue"
        }
    } // Qan.Navigable

    RowLayout {
        CheckBox {
            text: "Grid Visible"
            enabled: navigable.grid
            checked: navigable.grid ? navigable.grid.visible : false
            onCheckedChanged: navigable.grid.visible = checked
        }
        Label { text: "Grid Type:" }
        ComboBox {
            id: gridType
            textRole: "key"
            model: ListModel {
                ListElement { key: "Lines";  value: 25 }
                ListElement { key: "None"; value: 50 }
            }
            currentIndex: 0 // Default to "Lines"
            onActivated: {
                switch ( currentIndex ) {
                case 0: navigable.grid = lineGrid; break;
                case 2: navigable.grid = null; break;
                }
            }
        }
        Label { text: "Grid Scale:" }
        ComboBox {
            textRole: "key"
            model: ListModel {
                ListElement { key: "25";    value: 25 }
                ListElement { key: "50";    value: 50 }
                ListElement { key: "100";   value: 100 }
                ListElement { key: "150";   value: 150 }
            }
            currentIndex: 1 // Default to 100
            onActivated: {
                var gridScale = model.get(currentIndex).value
                if ( gridScale )
                    navigable.grid.gridScale = gridScale
            }
        }
        Label { Layout.leftMargin: 25; text: "Grid Major:" }
        SpinBox {
            from: 1;    to: 10
            enabled: navigable.grid
            value: navigable.grid ? navigable.grid.gridMajor : 0
            onValueModified: navigable.grid.gridMajor = value
        }
        Label { Layout.leftMargin: 25; text: "Point size:" }
        SpinBox {
            from: 1;    to: 10
            enabled: navigable.grid
            value: navigable.grid ? navigable.grid.gridWidth : 0
            onValueModified: navigable.grid.gridWidth = value
        }
    }
}
