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

import QtQuick 2.13
import QtQuick.Controls 2.15
import QtQuick.Layouts  1.3
import QtQuick.Controls.Material 2.1
import Qt.labs.platform 1.0 as Labs // FileDialog

Dialog {
    id: screenshotPopup

    // PUBLIC /////////////////////////////////////////////////////////////////
    title: qsTr('Export to Image')

    //! Graph view used to save graph screenshot image.
    property var graphView: undefined

    // PRIVATE ////////////////////////////////////////////////////////////////
    modal: true
    parent: Overlay.overlay
    x: (parent.width - width) / 2.
    y: (parent.height - height) / 2.
    focus: true
    ColumnLayout {
        anchors.fill: parent
        ColumnLayout{
            Layout.fillWidth: true
            Layout.fillHeight: false
            Label {
                text: qsTr('Output:')
                font.bold: true
            }
            RowLayout {
                TextField {
                    id: outputField
                    Layout.preferredWidth: 350
                    placeholderText: qsTr('Output file path')
                    selectByMouse: true
                    readOnly: true
                }
                Button {
                    flat: true
                    text: '...'
                    onClicked: selectOutputFd.open()
                    Labs.FileDialog {
                        id: selectOutputFd
                        title: qsTr('Select an output file')
                        acceptLabel: qsTr('Select')
                        fileMode: Labs.FileDialog.SaveFile
                        defaultSuffix: "png"
                        onAccepted: outputField.text = file
                        nameFilters: ["Image files (*.png)"]
                    }
                }
            }
        }  // ColumnLayout output
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: false
            Label {
                Layout.fillWidth: false
                text: 'Size:'
                font.bold: true
            }
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: false
                RadioButton {
                    id: zoom1
                    checked: true
                    text: qsTr("100%")
                }
                RadioButton {
                    id: zoom1dot5
                    text: qsTr("150%")
                }
                RadioButton {
                    id: zoom2
                    text: qsTr("200%")
                }
            }
        }  // ColumnLayout size
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: false
            Label {
                text: 'Borders:'
                font.bold: true
            }
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: false
                RadioButton {
                    id: border0
                    checked: true
                    text: qsTr("None")
                }
                RadioButton {
                    id: border20px
                    text: qsTr("20 px")
                }
                RadioButton {
                    id: border50px
                    text: qsTr("50 px")
                }
            }
        }  // ColumnLayout borders
    }  // ColumnLayout: dialog main layout
    footer: DialogButtonBox {
        //standardButtons: DialogButtonBox.Cancel
        Button {
            text: qsTr("Cancel")
            DialogButtonBox.buttonRole: DialogButtonBox.RejectRole
            focus: false
        }
        Button {
            text: qsTr("Save")
            DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole
            enabled: graphView && outputField.text !== ''
            focus: true
        }
    }
    onRejected: {
        console.error("CANCEL")
        close()
    }
    onAccepted: {
        console.error("outputField.text=" + outputField.text)
        if (graphView && outputField.text !== '') {
            let zoom = zoom1.checked ? 1 :
                                       zoom1dot5.checked ? 1.5:
                                                           zoom2.checked ? 2. : 1.
            let border = border0.checked ? 0. :
                                           border20px.checked ? 20. :
                                                                border50px.checked ? 50 : 0.
            graphView.grabGraphImage(outputField.text, zoom, border)
        }
        close()
    }
}  // Popup: screenshotPopup
