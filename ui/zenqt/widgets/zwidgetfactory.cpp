#include "zwidgetfactory.h"
#include "util/uihelper.h"
#include "widgets/zlinewidget.h"
#include "widgets/zlineedit.h"
#include "widgets/ztextedit.h"
#include "curvemap/zcurvemapeditor.h"
#include "dialog/zenoheatmapeditor.h"
#include "widgets/zcombobox.h"
#include "widgets/zlabel.h"
#include "style/zenostyle.h"
#include "widgets/ColorEditor.h"
#include "model/graphsmanager.h"
#include "model/curvemodel.h"
#include "widgets/zveceditor.h"
#include "widgets/zcomboboxitemdelegate.h"
#include "variantptr.h"
#include "zassert.h"
#include "zspinboxslider.h"
#include "zdicttableview.h"
#include "nodeeditor/gv/zitemfactory.h"
#include "widgets/zpathedit.h"
#include "util/uihelper.h"
#include "util/jsonhelper.h"
#include "widgets/zcodeeditor.h"

namespace zenoui
{
    QWidget* createWidget(
        const QModelIndex& nodeIdx,
        const QVariant& value,
        zeno::ParamControl ctrl,
        const zeno::ParamType type,
        CallbackCollection cbSet,
        const zeno::ControlProperty& controlProps
    )
    {
        switch (ctrl)
        {
            case zeno::Lineedit:
            {
                QString text = UiHelper::variantToString(value);
                ZLineEdit *pLineEdit = new ZLineEdit(text);

                pLineEdit->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                pLineEdit->setProperty("cssClass", "zeno2_2_lineedit");
                pLineEdit->setNumSlider(UiHelper::getSlideStep("", type));
                pLineEdit->setNodeIdx(nodeIdx);
                QObject::connect(pLineEdit, &ZLineEdit::editingFinished, [=]() {
                    // be careful about the dynamic type.
                    QString text = pLineEdit->text();
                    const QVariant& newValue = UiHelper::parseStringByType(text, type);
                    cbSet.cbEditFinished(newValue);
                    });
                return pLineEdit;
            }
            case zeno::Checkbox:
            {
                QCheckBox* pCheckbox = new QCheckBox;
                pCheckbox->setCheckState(value.toBool() ? Qt::Checked : Qt::Unchecked);
                QObject::connect(pCheckbox, &QCheckBox::stateChanged, [=](int state) {
                    cbSet.cbEditFinished(state);
                });
                return pCheckbox;
            }
            case zeno::ReadPathEdit:
            case zeno::WritePathEdit:
            case zeno::DirectoryPathEdit:
            {
                ZPathEdit *pathLineEdit = new ZPathEdit(value.toString(), ctrl);
                pathLineEdit->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                pathLineEdit->setProperty("control", ctrl);
                
                QObject::connect(pathLineEdit, &ZLineEdit::textEditFinished, [=]() {
                    cbSet.cbEditFinished(pathLineEdit->text());
                });
                return pathLineEdit;
            }
            case zeno::Multiline:
            {
                ZTextEdit* pTextEdit = new ZTextEdit;
                pTextEdit->setNodeIdx(nodeIdx);
                pTextEdit->setFrameShape(QFrame::NoFrame);
                pTextEdit->setProperty("cssClass", "proppanel");
                pTextEdit->setProperty("control", ctrl);
                QFont font = QApplication::font();
                font.setPointSize(9);
                pTextEdit->setFont(font);
                pTextEdit->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);

                QTextCharFormat format;
                format.setFont(font);
                pTextEdit->setCurrentFont(font);
                pTextEdit->setText(value.toString());

                QPalette pal = pTextEdit->palette();
                pal.setColor(QPalette::Base, QColor(61, 61, 61));
                pTextEdit->setPalette(pal);

                QObject::connect(pTextEdit, &ZTextEdit::editFinished, [=]() {
                    cbSet.cbEditFinished(pTextEdit->toPlainText());
                });
                return pTextEdit;
            }
            //case CONTROL_PYTHON_EDITOR:
            //{
            //    ZPythonEditor* pythonEditor = new ZPythonEditor(UiHelper::variantToString(value));
            //    pythonEditor->setFixedHeight(ZenoStyle::dpiScaled(250));

            //    QObject::connect(pythonEditor, &ZPythonEditor::editingFinished, [=]() {
            //        const QString& newValue = pythonEditor->text();
            //    cbSet.cbEditFinished(newValue);
            //    });
            //    return pythonEditor;
            //}
            case zeno::Heatmap:
            {
                QPushButton* pBtn = new QPushButton("Edit Heatmap");
                pBtn->setProperty("cssClass", "proppanel");
                QObject::connect(pBtn, &QPushButton::clicked, [=]() {
                    QString val = cbSet.cbGetIndexData().toString();
                    ZenoHeatMapEditor editor(val);
                    editor.exec();
                    QString newVal = editor.colorRamps();
                    cbSet.cbEditFinished(QVariant::fromValue(newVal));
                });
                return pBtn;
            }
            case zeno::ColorVec:
            {
                QColor currentColor;
                if (ctrl == zeno::ColorVec) {
                    auto colorVec = value.value<UI_VECTYPE>();
                    colorVec.resize(3);
                    currentColor = QColor::fromRgbF(colorVec[0], colorVec[1], colorVec[2]);
                }
                QPushButton *pBtn = new QPushButton;
                pBtn->setFixedSize(ZenoStyle::dpiScaled(100), ZenoStyle::dpiScaled(30));
                pBtn->setStyleSheet(QString("background-color:%1; border:0;").arg(currentColor.name()));
                QObject::connect(pBtn, &QPushButton::clicked, [=]() {
                    QColor color = ColorEditor::getColor(pBtn->palette().window().color());
                    if (color.isValid()) 
                    {
                        pBtn->setStyleSheet(QString("background-color:%1; border:0;").arg(color.name()));
                        if (ctrl == zeno::ColorVec) {
                            UI_VECTYPE colorVec(3);
                            color.getRgbF(&colorVec[0], &colorVec[1], &colorVec[2]);
                            cbSet.cbEditFinished(QVariant::fromValue<UI_VECTYPE>(colorVec));
                        }
                    }
                });
                return pBtn;
            }
            case zeno::Vec2edit:
            case zeno::Vec3edit:
            case zeno::Vec4edit:
            {
                int dim = -1;
                bool bFloat = false;
                if (type == zeno::Param_Vec2i || type == zeno::Param_Vec2f)
                {
                    dim = 2;
                    bFloat = type == zeno::Param_Vec2f;
                }
                else if (type == zeno::Param_Vec3i || type == zeno::Param_Vec3f)
                {
                    dim = 3;
                    bFloat = type == zeno::Param_Vec3f;
                }
                else if (type == zeno::Param_Vec4i || type == zeno::Param_Vec4f)
                {
                    dim = 4;
                    bFloat = type == zeno::Param_Vec4f;
                }

                ZVecEditor* pVecEdit = new ZVecEditor(value, bFloat, dim, "zeno2_2_lineedit");
                pVecEdit->setNodeIdx(nodeIdx);
                pVecEdit->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                QObject::connect(pVecEdit, &ZVecEditor::editingFinished, [=]() {
                    const QVariant &newValue = pVecEdit->vec();
                    cbSet.cbEditFinished(newValue);
                });
                return pVecEdit;
            }
            case zeno::Combobox:
            {
                QStringList items;
                if (controlProps.items.has_value())
                {
                    for (auto item : controlProps.items.value())
                        items.push_back(QString::fromStdString(item));
                }

                ZComboBox *pComboBox = new ZComboBox;
                pComboBox->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                pComboBox->addItems(items);
                pComboBox->setCurrentText(value.toString());
                pComboBox->setEditable(true);
                pComboBox->setItemDelegate(new ZComboBoxItemDelegate2(pComboBox));

                QObject::connect(pComboBox, &ZComboBox::_textActivated, [=](const QString& text) {
                    cbSet.cbEditFinished(text);
                });
                return pComboBox;
            }
            case zeno::CurveEditor:
            {
                QPushButton* pBtn = new QPushButton("Edit Curve");
                pBtn->setProperty("cssClass", "proppanel");
                QObject::connect(pBtn, &QPushButton::clicked, [=]() {
                    ZCurveMapEditor* pEditor = new ZCurveMapEditor(true);
                    pEditor->setAttribute(Qt::WA_DeleteOnClose);

                    QObject::connect(pEditor, &ZCurveMapEditor::finished, [=](int result) {
                        QString newVal = JsonHelper::dumpCurves(pEditor->curves());
                        cbSet.cbEditFinished(newVal);
                    });

                    CURVES_DATA curves;
                    if (cbSet.cbGetIndexData)
                    {
                        const QString &str = cbSet.cbGetIndexData().toString();
                        curves = JsonHelper::parseCurves(str);
                    }
                    pEditor->addCurves(curves);
                    pEditor->exec();
                });
                return pBtn;
            }
            case zeno::Slider:
            {
                QSlider* pSlider = new QSlider(Qt::Horizontal);
                pSlider->setStyleSheet(ZenoStyle::dpiScaleSheet("\
                    QSlider::groove:horizontal {\
                        height: 4px;\
                        background: #707D9C;\
                    }\
                    \
                    QSlider::handle:horizontal {\
                        background: #DFE2E5;\
                        width: 6px;\
                        margin: -8px 0;\
                    }\
                    QSlider::add-page:horizontal {\
                        background: rgb(61,61,61);\
                    }\
                    \
                    QSlider::sub-page:horizontal {\
                        background: #707D9C;\
                    }\
                "));
                pSlider->setValue(value.toInt());

                SLIDER_INFO sliderInfo;
                
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }
                pSlider->setSingleStep(sliderInfo.step);
                pSlider->setRange(sliderInfo.min, sliderInfo.max);

                QObject::connect(pSlider, &QSlider::valueChanged, [=](int value) {
                    cbSet.cbEditFinished(value);
                });

                QObject::connect(pSlider, &QSlider::sliderPressed, [=]() {
                    QRect rc = pSlider->rect();
                    QPoint br = pSlider->mapToGlobal(rc.bottomRight());
                    QPoint pos = QCursor::pos();
                    pos.setY(br.y());
                    QToolTip::showText(pos, QString("%1").arg(pSlider->value()), nullptr);
                });

                QObject::connect(pSlider, &QSlider::sliderMoved, [=](int value) {
                    QRect rc = pSlider->rect();
                    QPoint br = pSlider->mapToGlobal(rc.bottomRight());
                    QPoint pos = QCursor::pos();
                    pos.setY(br.y());
                    QToolTip::showText(pos, QString("%1").arg(value), nullptr);
                });
                return pSlider;
            }
            case zeno::SpinBox:
            {
                QSpinBox* pSpinBox = new QSpinBox;
                pSpinBox->setProperty("cssClass", "control");
                pSpinBox->setAlignment(Qt::AlignCenter);
                pSpinBox->setValue(value.toInt());
                pSpinBox->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                SLIDER_INFO sliderInfo;
                
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }
                pSpinBox->setSingleStep(sliderInfo.step);
                pSpinBox->setRange(sliderInfo.min, sliderInfo.max);
                QObject::connect(pSpinBox, static_cast<void (QSpinBox::*)(int)>(& QSpinBox::valueChanged),[=](int value) { 
					cbSet.cbEditFinished(value);
				});
                return pSpinBox;
            }
            case zeno::DoubleSpinBox: {
                QDoubleSpinBox *pSpinBox = new QDoubleSpinBox;
                pSpinBox->setProperty("cssClass", "control");
                pSpinBox->setAlignment(Qt::AlignCenter);
                pSpinBox->setValue(value.toDouble());
                pSpinBox->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                SLIDER_INFO sliderInfo;
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }
                pSpinBox->setSingleStep(sliderInfo.step);
                pSpinBox->setRange(sliderInfo.min, sliderInfo.max);
                QObject::connect(pSpinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),[=](double value) {
                    cbSet.cbEditFinished(value); 
                });
                return pSpinBox;
            }
            case zeno::SpinBoxSlider:
            {
                ZSpinBoxSlider* pSlider = new ZSpinBoxSlider;
                SLIDER_INFO sliderInfo;
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }
                pSlider->setSingleStep(sliderInfo.step);
                pSlider->setRange(sliderInfo.min, sliderInfo.max);
                pSlider->setValue(value.toInt());
                QObject::connect(pSlider, &ZSpinBoxSlider::valueChanged, [=](int value) {
                    cbSet.cbEditFinished(value);
                });
                return pSlider;
            }
            case zeno::CodeEditor:
            {
                ZCodeEditor* pCodeEditor = new ZCodeEditor(value.toString());
                QObject::connect(pCodeEditor, &ZCodeEditor::editFinished, [=](const QString& text) {
                    cbSet.cbEditFinished(text);
                });
                return pCodeEditor;
            }
            default:
                return nullptr;
        }
        return nullptr;
    }

    bool isMatchControl(zeno::ParamControl ctrl, QWidget* pControl)
    {
        if (!pControl)
            return false;

        switch (ctrl)
        {
        case zeno::Lineedit:    return qobject_cast<ZLineEdit*>(pControl) != nullptr;    //be careful type changed.
        case zeno::ReadPathEdit:
        case zeno::WritePathEdit:
        case zeno::DirectoryPathEdit:
            return qobject_cast<ZLineEdit*>(pControl) != nullptr;
        case zeno::Checkbox:    return qobject_cast<QCheckBox*>(pControl) != nullptr;
        case zeno::Vec2edit:
        case zeno::Vec3edit:
        case zeno::Vec4edit:    return qobject_cast<ZVecEditor*>(pControl) != nullptr;
        case zeno::Combobox:    return qobject_cast<QComboBox*>(pControl) != nullptr;
        case zeno::Multiline:   return qobject_cast<ZTextEdit*>(pControl) != nullptr;
        case zeno::CurveEditor: //only support a button to emit dialog of curve
        case zeno::ColorVec:       return qobject_cast<QPushButton*>(pControl) != nullptr;
        }
    }

    void updateValue(QWidget* pControl, const QVariant& value)
    {
        if (ZLineEdit* pLineEdit = qobject_cast<ZLineEdit*>(pControl))
        {
            pLineEdit->setText(value.toString());
        }
        else if (QCheckBox* pCheckbox = qobject_cast<QCheckBox*>(pControl))
        {
            pCheckbox->setCheckState(value.toBool() ? Qt::Checked : Qt::Unchecked);
        }
        else if (ZVecEditor* pVecEditor = qobject_cast<ZVecEditor*>(pControl))
        {
            pVecEditor->setVec(value, pVecEditor->isFloat());
        }
        else if (ZTextEdit* pTextEdit = qobject_cast<ZTextEdit*>(pControl))
        {
            pTextEdit->setText(value.toString());
        }
        else if (QComboBox* pCombobox = qobject_cast<QComboBox*>(pControl))
        {
            pCombobox->setCurrentText(value.toString());
        }
    }

}