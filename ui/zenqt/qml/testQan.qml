/*
 Copyright (c) 2008-2023, Benoit AUTHEMAN All rights reserved.

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

import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0
import QtQuick.Controls.Styles   1.4
import Qt.labs.settings 1.1

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import Zeno 1.0 as Zen
import "./view"
import "./container/TabView"


Zen.GraphsTotalView {
    id: totalview
    visible: true
    width: 1280; height: 720
    //anchors.fill: parent
    graphsMgr: graphsmanager

    Material.theme: Material.Dark

    StackLayout {
        id: welcomepage_or_editor
        currentIndex: 0
        anchors.fill: parent

        WelcomePage {
            Layout.fillHeight: true
            Layout.fillWidth: true
            graphs: totalview
        }

        /*//如果提前初始化，那么主图的组件也会提前初始化，然而这时候主图model还没初始化（因为还没有打开/新建文件）
        ZGraphsView {
            id: graphsallview
            Layout.fillHeight: true
            Layout.fillWidth: true
        }
        */
    }

    onModelInited: function() {
        //动态添加ZGraphsView，这样主图模型就已经初始化了
        const graphsview_comp = Qt.createComponent("qrc:/ZGraphsView.qml")
        const newgraphsview = graphsview_comp.createObject(welcomepage_or_editor, { id: "graphsallview" })
        welcomepage_or_editor.currentIndex = 1
        graphsMgr.currentPath = "/main"
    }

    onFileClosed: function() {
        if (welcomepage_or_editor.currentIndex == 1) {
            welcomepage_or_editor.currentIndex = 0
            welcomepage_or_editor.children[1].destroy()
        }
    }
}

