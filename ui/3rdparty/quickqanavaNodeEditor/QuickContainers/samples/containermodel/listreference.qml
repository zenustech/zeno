/*
    This file is part of QuickProperties2 library.

    Copyright (C) 2016  Benoit AUTHEMAN

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

import QtQuick          2.7
import QtQuick.Controls 2.0
import QtQuick.Layouts  1.3

import QuickContainers  1.0

import ContainerModelSample 1.0

Item {
    Pane {
        id: dummies1Pane
        anchors.top: parent.top; anchors.left: parent.left
        anchors.leftMargin: 25; anchors.topMargin: 15
        width: 250; height: 300
        ColumnLayout {
            anchors.fill: parent
            spacing: 15
            Text {  Layout.maximumWidth: parent.width; font.bold: true; wrapMode: Text.WrapAnywhere
                    text:"qps::ContainerModel<QVector,Dummy*>" }
            RowLayout {
                Button {
                    text:"+"
                    onClicked: {
                        dummies1.listReference.append(42);
                    }
                }
                Button {
                    text:"-"
                    onClicked: {
                        dummies1.listReference.remove(m1List.currentItem)
                        //m1.remove(m1List.currentItem);
                    }
                }
            }
            Text { text:"Item count=" + dummies1.itemCount }
            ListView {
                Layout.fillWidth: true; Layout.fillHeight: true
                model: dummies1
                highlightFollowsCurrentItem: true
                delegate: Text { text: itemData.label + " " + itemData.number }
            }
            ComboBox {
                Layout.fillWidth: true; Layout.fillHeight: false
                textRole: "itemLabel"
                model: dummies1
            }
        }
    } // Pane m1

    Component {
        id: dummyComponent
        Dummy { }
    }
    Pane {
        id: dummies2Pane
        anchors.top: parent.top; anchors.left: dummies1Pane.right
        anchors.leftMargin: 25; anchors.topMargin: 15
        width: 250; height: 300
        ColumnLayout {
            anchors.fill: parent
            spacing: 15
            Text {  Layout.maximumWidth: parent.width; font.bold: true; wrapMode: Text.WrapAnywhere
                    text:"qps::ContainerModel<QVector,Dummy*>" }
            RowLayout {
                Button {
                    text:"+"
                    onClicked: {
                        var dummy = dummyComponent.createObject();
                        console.debug( "dummy=" + dummy );
                        dummy.label = "Test"
                        dummy.number = 42
                        dummies2.listReference.append(dummy);
                    }
                }
                Button {
                    text:"-"
                    onClicked: {
                        console.debug("Removing dummies2List.currentItem=" + dummies2List.currentItem)
                        dummies2.listReference.remove(dummies2List.currentItem)
                    }
                }
            }
            Text { text:"Item count=" + dummies2.itemCount }
            ListView {
                highlightFollowsCurrentItem : false
                id: dummies2List
                Layout.fillWidth: true; Layout.fillHeight: true
                model: dummies2
                delegate: Text { text: itemData.label + " " + itemData.number }
            }
            ComboBox {
                Layout.fillWidth: true; Layout.fillHeight: false
                textRole: "itemLabel"
                model: dummies2
            }
        }
    } // Pane m1
}

