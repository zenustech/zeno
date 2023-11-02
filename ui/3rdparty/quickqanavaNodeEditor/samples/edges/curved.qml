/*
 Copyright (c) 2008-2020, Benoit AUTHEMAN All rights reserved.

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

import Qt.labs.platform     1.1

import QuickQanava          2.0 as Qan
import "qrc:/QuickQanava"   as Qan

Item {
    anchors.fill: parent
    Qan.GraphView {
        id: graphView
        anchors.fill: parent
        navigable   : true
        graph: Qan.Graph {
            id: graph
            connectorEnabled: true
            Component.onCompleted: {
                var c = graph.insertNode()
                c.label = "C"; c.item.x = 350; c.item.y = 200


                var tl = graph.insertNode()
                tl.label = "TL"; tl.item.x = 50; tl.item.y = 50
                graph.insertEdge(c, tl);

                var tml = graph.insertNode()
                tml.label = "TML"; tml.item.x = 200; tml.item.y = 50
                graph.insertEdge(c, tml);

                var t = graph.insertNode()
                t.label = "T"; t.item.x = 350; t.item.y = 50
                graph.insertEdge(c, t);

                var tmr = graph.insertNode()
                tmr.label = "TMR"; tmr.item.x = 500; tmr.item.y = 50
                graph.insertEdge(c, tmr);

                var tr = graph.insertNode()
                tr.label = "TR"; tr.item.x = 650; tr.item.y = 50
                graph.insertEdge(c, tr);


                var bl = graph.insertNode()
                bl.label = "BL"; bl.item.x = 50; bl.item.y = 350
                graph.insertEdge(c, bl);

                var bml = graph.insertNode()
                bml.label = "BML"; bml.item.x = 200; bml.item.y = 350
                graph.insertEdge(c, bml);

                var b = graph.insertNode()
                b.label = "B"; b.item.x = 350; b.item.y = 350
                graph.insertEdge(c, b);

                var bmr = graph.insertNode()
                bmr.label = "BMR"; bmr.item.x = 500; bmr.item.y = 350
                graph.insertEdge(c, bmr);

                var br = graph.insertNode()
                br.label = "BR"; br.item.x = 650; br.item.y = 350
                graph.insertEdge(c, br);


                var l = graph.insertNode()
                l.label = "L"; l.item.x = 50; l.item.y = 200
                graph.insertEdge(c, l);

                var r = graph.insertNode()
                r.label = "R"; r.item.x = 650; r.item.y = 200
                graph.insertEdge(c, r);

                graph.setConnectorSource(c)

                var x = 850; var y = 80
                generateTestPortLayout(x, y,             Qan.NodeItem.Left, Qan.NodeItem.Left);
                generateTestPortLayout(x + 420, y,       Qan.NodeItem.Left, Qan.NodeItem.Top);
                generateTestPortLayout(x, y + 200,       Qan.NodeItem.Left, Qan.NodeItem.Right);
                generateTestPortLayout(x + 420, y + 200, Qan.NodeItem.Left, Qan.NodeItem.Bottom);

                x = 1700; y = 80
                generateTestPortLayout(x, y,             Qan.NodeItem.Top, Qan.NodeItem.Left);
                generateTestPortLayout(x + 420, y,       Qan.NodeItem.Top, Qan.NodeItem.Top);
                generateTestPortLayout(x, y + 200,       Qan.NodeItem.Top, Qan.NodeItem.Right);
                generateTestPortLayout(x + 420, y + 200, Qan.NodeItem.Top, Qan.NodeItem.Bottom);

                x = 850; y = 400
                generateTestPortLayout(x, y,             Qan.NodeItem.Right, Qan.NodeItem.Left);
                generateTestPortLayout(x + 420, y,       Qan.NodeItem.Right, Qan.NodeItem.Top);
                generateTestPortLayout(x, y + 200,       Qan.NodeItem.Right, Qan.NodeItem.Right);
                generateTestPortLayout(x + 420, y + 200, Qan.NodeItem.Right, Qan.NodeItem.Bottom);

                x = 1700; y = 400
                generateTestPortLayout(x, y,             Qan.NodeItem.Bottom, Qan.NodeItem.Left);
                generateTestPortLayout(x + 420, y,       Qan.NodeItem.Bottom, Qan.NodeItem.Top);
                generateTestPortLayout(x, y + 200,       Qan.NodeItem.Bottom, Qan.NodeItem.Right);
                generateTestPortLayout(x + 420, y + 200, Qan.NodeItem.Bottom, Qan.NodeItem.Bottom);

                x = 80; y = 600
                generateTestPortNodeLayout(x, y,             Qan.NodeItem.Left );
                generateTestPortNodeLayout(x + 420, y,       Qan.NodeItem.Top );
                generateTestPortNodeLayout(x, y + 200,       Qan.NodeItem.Right );
                generateTestPortNodeLayout(x + 420, y + 200, Qan.NodeItem.Bottom );
            }

            function generateTestPortNodeLayout(x, y, srcPortType ) {
                // SRC/DST horizontally aligned
                var s = graph.insertNode()
                s.label = "S1"; s.item.x = x; s.item.y = y
                var sp1 = graph.insertPort(s, srcPortType);
                sp1.label = "OUT#1"

                var d = graph.insertNode()
                d.label = "D1"; d.item.x = x + 200; d.item.y = y

                var e = graph.insertEdge(s, d);
                graph.bindEdgeSource(e, sp1)
            }

            function generateTestPortLayout(x, y, srcPortType, dstPortType) {
                // SRC/DST horizontally aligned
                var s = graph.insertNode()
                s.label = "S1"; s.item.x = x; s.item.y = y
                var sp1 = graph.insertPort(s, srcPortType);
                sp1.label = "OUT#1"

                var d = graph.insertNode()
                d.label = "D1"; d.item.x = x + 200; d.item.y = y
                var dp1 = graph.insertPort(d, dstPortType);
                dp1.label = "IN#1"

                var e = graph.insertEdge(s, d);
                graph.bindEdgeSource(e, sp1)
                //graph.bindEdgeDestination(e, sp1)
                graph.bindEdgeDestination(e, dp1)

                // SRC/DST vertically aligned
                /*var s = graph.insertNode()
            s.label = "S1"; s.item.x = x + 450; s.item.y = y
            var sp1 = graph.insertPort(s, srcPortType);
            sp1.label = "OUT#1"

            var d = graph.insertNode()
            d.label = "D1"; d.item.x = x + 450; d.item.y = y + 200
            var dp1 = graph.insertPort(d, dstPortType);
            dp1.label = "IN#1"

            var e = graph.insertEdge(s, d);
            graph.bindEdgeSource(e, sp1)
            graph.bindEdgeDestination(e, sp1)
            graph.bindEdgeDestination(e, dp1)

            // SRC bottom left / DST top right
            var s = graph.insertNode()
            s.label = "S"; s.item.x = x; s.item.y = y + 150
            var sp1 = graph.insertPort(s, srcPortType);
            sp1.label = "OUT#1"

            var d = graph.insertNode()
            d.label = "D1"; d.item.x = x + 300; d.item.y = y + 300
            var dp1 = graph.insertPort(d, dstPortType);
            dp1.label = "IN#1"

            var e = graph.insertEdge(s, d);
            graph.bindEdgeSource(e, sp1)
            graph.bindEdgeDestination(e, sp1)
            graph.bindEdgeDestination(e, dp1)

            // SRC top left / DST bottom right
            var s = graph.insertNode()
            s.label = "S"; s.item.x = x + 300; s.item.y = y + 400
            var sp1 = graph.insertPort(s, srcPortType);
            sp1.label = "OUT#1"

            var d = graph.insertNode()
            d.label = "D1"; d.item.x = x; d.item.y = y + 550
            var dp1 = graph.insertPort(d, dstPortType);
            dp1.label = "IN#1"

            var e = graph.insertEdge(s, d);
            graph.bindEdgeSource(e, sp1)
            graph.bindEdgeDestination(e, sp1)
            graph.bindEdgeDestination(e, dp1)*/
            }
        }
    }  // Qan.GraphView

    ColorDialog {
        id: edgeStyleColorDialog
        title: "Edge color"
        onAccepted: { defaultEdgeStyle.lineColor = color; }
    }

    Pane {
        id: edgeStyleEditor
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 15
        anchors.right: parent.right
        anchors.rightMargin: 15
        padding: 0
        Frame {
            ColumnLayout {
                Label {
                    Layout.margins: 3; text: "Edge Style:"
                    font.bold: true; horizontalAlignment: Text.AlignLeft
                }
                RowLayout {
                    Layout.margins: 2
                    Label { text:"Edge type:" }
                    Item { Layout.fillWidth: true }
                    ComboBox {
                        model: ["Straight", "Curved"]
                        enabled: defaultEdgeStyle !== undefined
                        currentIndex: defaultEdgeStyle.lineType === Qan.EdgeStyle.Straight ? 0 : 1
                        onActivated: {
                            if (index == 0 )
                                defaultEdgeStyle.lineType = Qan.EdgeStyle.Straight
                            else if ( index == 1 )
                                defaultEdgeStyle.lineType = Qan.EdgeStyle.Curved
                        }
                    }
                } // RowLayout: edgeType
                RowLayout {
                    Layout.margins: 2
                    Label { text:"Src Shape:" }
                    Item { Layout.fillWidth: true }
                    ComboBox {
                        model: ["None", "Arrow", "Open Arrow", "Circle", "Open Circle", "Rect", "Open Rect"]
                        currentIndex: defaultEdgeStyle.lineType === Qan.EdgeStyle.Straight ? 0 : 1
                        onActivated: {
                            var shape = [Qan.EdgeStyle.None, Qan.EdgeStyle.Arrow, Qan.EdgeStyle.ArrowOpen, Qan.EdgeStyle.Circle, Qan.EdgeStyle.CircleOpen, Qan.EdgeStyle.Rect, Qan.EdgeStyle.RectOpen]
                            defaultEdgeStyle.srcShape = shape[index]
                        }
                    }
                } // RowLayout: srcShape
                RowLayout {
                    Layout.margins: 2
                    Label { text:"Dst shape:" }
                    Item { Layout.fillWidth: true }
                    ComboBox {
                        model: ["None", "Arrow", "Open Arrow", "Circle", "Open Circle", "Rect", "Open Rect"]
                        currentIndex: 1
                        onActivated: {
                            var shape = [Qan.EdgeStyle.None, Qan.EdgeStyle.Arrow, Qan.EdgeStyle.ArrowOpen, Qan.EdgeStyle.Circle, Qan.EdgeStyle.CircleOpen, Qan.EdgeStyle.Rect, Qan.EdgeStyle.RectOpen]
                            defaultEdgeStyle.dstShape = shape[index]
                        }
                    }
                } // RowLayout: dstShape
                RowLayout {
                    Layout.margins: 2
                    Label { text:"Line color:" }
                    Item { Layout.fillWidth: true }
                    Rectangle {
                        Layout.preferredWidth: 25; Layout.preferredHeight: 25;
                        color: defaultEdgeStyle.lineColor; radius: 5;
                        border.width: 1; border.color: Qt.lighter(defaultEdgeStyle.lineColor)
                    }
                    ToolButton {
                        Layout.preferredHeight: 30; Layout.preferredWidth: 30
                        text: "..."
                        onClicked: {
                            edgeStyleColorDialog.color = defaultEdgeStyle.lineColor
                            edgeStyleColorDialog.open();
                        }
                    }
                } // RowLayout: lineColor
                ColumnLayout {
                    Layout.margins: 2
                    Label { text:"Line width:" }
                    SpinBox {
                        value: defaultEdgeStyle.lineWidth
                        from: 1; to: 7
                        onValueModified: defaultEdgeStyle.lineWidth = value
                    }
                } // RowLayout: lineWidth
                ColumnLayout {
                    Layout.margins: 2
                    Label { text:"Arrow size:" }
                    SpinBox {
                        value: defaultEdgeStyle.arrowSize
                        from: 1; to: 7
                        onValueModified: defaultEdgeStyle.arrowSize = value
                    }
                } // RowLayout: lineWidth

                ColumnLayout {
                    Layout.margins: 2
                    CheckBox {
                        id: dashed
                        text: "Dashed"
                        checked: defaultEdgeStyle.dashed
                        onClicked: {
                            if (defaultEdgeStyle)
                                defaultEdgeStyle.dashed = checked
                        }
                    }
                    ComboBox {
                        enabled: dashed.checked
                        model: ["Dash", "Dot", "Dash Dot"]
                        //currentIndex: defaultEdgeStyle.lineType === Qan.EdgeStyle.Straight ? 0 : 1
                        onActivated: {
                            if (index == 0 )
                                defaultEdgeStyle.dashPattern = [2,2]
                            else if ( index == 1 )
                                defaultEdgeStyle.dashPattern = [1,1]
                            else if ( index == 2 )
                                defaultEdgeStyle.dashPattern = [2,2,4,2]
                        }
                    }
                } // ColumneLayout: dashed

                ColumnLayout {
                    Layout.margins: 2
                    Label { text:"Arrow size:" }
                    SpinBox {
                        value: defaultEdgeStyle.arrowSize
                        from: 1; to: 15
                        onValueModified: defaultEdgeStyle.arrowSize = value
                    }
                } // RowLayout: arrowSize
            } // ColumnLayout
        }
    } // edgeStyleEditor
} // Root item


