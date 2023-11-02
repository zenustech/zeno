/*
 Copyright (c) 2008-2022, Benoit AUTHEMAN All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the author or Destrat.io nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL AUTHOR BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

import QtQuick                   2.8
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan

ApplicationWindow {
    id: window
    visible: true
    width: 1280; height: 720
    title: "Custom nodes sample"
    Pane { anchors.fill: parent }
    ColumnLayout {
        anchors.fill: parent
        TabBar {
            id: tabBar
            Layout.preferredWidth: 450; Layout.fillHeight: false
            TabButton { text: qsTr("Default Nodes") }
            TabButton { text: qsTr("Custom Nodes") }
        }
        StackLayout {
            clip: true
            Layout.fillWidth: true; Layout.fillHeight: true
            currentIndex: tabBar.currentIndex
            Item { Loader { anchors.fill: parent; source: "qrc:/default.qml"} }
            Item { Loader { anchors.fill: parent; source: "qrc:/custom.qml"} }
        }
    }
    RowLayout {
        anchors.top: parent.top;    anchors.right: parent.right
        CheckBox {
            text: qsTr("Dark")
            checked: ApplicationWindow.contentItem.Material.theme === Material.Dark
            onClicked: ApplicationWindow.contentItem.Material.theme = checked ? Material.Dark : Material.Light
        }
    }
}

