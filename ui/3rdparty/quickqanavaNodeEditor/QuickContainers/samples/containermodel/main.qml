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

Item {
    id: mainVindow
    anchors.fill: parent
    TabBar {
        id: tabBar
        Layout.fillWidth: true; Layout.fillHeight: false
        TabButton { text: "Containers" }
        TabButton { text: "List Reference" }
    }
    StackLayout {
        Layout.fillWidth: true; Layout.fillHeight: true
        currentIndex: tabBar.currentIndex
        Item {
            Loader { anchors.fill: parent; anchors.topMargin: tabBar.height; source: "qrc:/containermodel.qml"}
        }
        Item {
            Loader { anchors.fill: parent; anchors.topMargin: tabBar.height; source: "qrc:/listreference.qml"}
        }
    }
}

