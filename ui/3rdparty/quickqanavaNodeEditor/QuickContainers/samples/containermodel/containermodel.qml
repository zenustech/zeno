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
        id: m1Pane
        anchors.top: parent.top; anchors.left: parent.left
        anchors.leftMargin: 25; anchors.topMargin: 15
        width: 270; height: 300
        ColumnLayout {
            anchors.fill: parent
            spacing: 15
            Text {  Layout.maximumWidth: parent.width; font.bold: true; wrapMode: Text.WrapAnywhere
                    text:"qps::ContainerModel<QVector,int>" }
            Text { text:"Item count=" + ints.itemCount }
            Frame {
                Layout.fillWidth: true; Layout.fillHeight: true
                ListView {
                    anchors.fill: parent
                    model: ints
                    highlightFollowsCurrentItem: true
                    delegate: Text { text: "int=" + itemData }
                }
            }
            ComboBox {
                Layout.fillWidth: true; Layout.fillHeight: false
                textRole: "itemLabel"
                model: ints
            }
        }
    } // Pane ints

    Pane {
        id: m2Pane
        anchors.top: parent.top; anchors.left: m1Pane.right
        anchors.leftMargin: 25; anchors.topMargin: 15
        width: 300; height: 300
        ColumnLayout {
            anchors.fill: parent
            spacing: 15
            Text {  id: title; Layout.maximumWidth: parent.width; font.bold: true; wrapMode: Text.WrapAnywhere
                    text:"qps::ContainerModel<QVector,Dummy*>\n (Dummy is a QObject)" }
            Text { text:"Item count=" + dummies.itemCount }
            Button {
                Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
                text: "Modify"
                onClicked: {
                    var itemObject = dummies.listReference.at(0)
                    itemObject.label = itemObject.label + "modified"
                    itemObject.number = itemObject.number + 1
                }
            }
            Frame {
                Layout.fillWidth: true; Layout.fillHeight: true
                ListView {
                    anchors.fill: parent
                    model: dummies
                    delegate: Text { text: itemData.label + " " + itemData.number }
                }
            }
            ComboBox {
                Layout.fillWidth: true; Layout.fillHeight: false
                textRole: "itemLabel"
                model: dummies
            }
        }
    } // Pane dummies
}

