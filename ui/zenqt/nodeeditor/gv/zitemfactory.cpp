#include "zitemfactory.h"
#include "util/uihelper.h"
#include "util/ztfutil.h"
#include "nodeeditor/gv/zlineedititem.h"
#include "widgets/ColorEditor.h"
#include "model/curvemodel.h"
#include "zveceditoritem.h"
#include "style/zenostyle.h"
#include "dialog/zenoheatmapeditor.h"
#include "curvemap/zcurvemapeditor.h"
#include "variantptr.h"
#include "zassert.h"
#include "zgraphicstextitem.h"
#include "util/uihelper.h"
#include "util/ztfutil.h"
#include "util/jsonhelper.h"

/*tmp macro*/
//#define ENABLE_WIDGET_LINEEDIT


namespace zenoui
{
    static QString getOpenFileName(
        const QString& caption,
        const QString& dir,
        const QString& filter
    ) {
        QString path = QFileDialog::getOpenFileName(nullptr, caption, dir, filter);
#if 0 // cannot work for now, wait for StringEval to be integrated into string param edit (luzh job)
        QSettings settings("ZenusTech", "Zeno");
        QVariant nas_loc_v = settings.value("nas_loc");
        path.replace('\\', '/');
        if (!nas_loc_v.isNull()) {
            QString nas = nas_loc_v.toString();
            if (!nas.isEmpty()) {
                nas.replace('\\', '/');
                path.replace(nas, "$NASLOC");
            }
        }
#endif
        return path;
    }

    static QString getSaveFileName(
        const QString& caption,
        const QString& dir,
        const QString& filter
    ) {
        QString path = QFileDialog::getSaveFileName(nullptr, caption, dir, filter);
#if 0 // cannot work for now, wait for StringEval to be integrated into string param edit (luzh job)
        QSettings settings("ZenusTech", "Zeno");
        QVariant nas_loc_v = settings.value("nas_loc");
        path.replace('\\', '/');
        if (!nas_loc_v.isNull()) {
            QString nas = nas_loc_v.toString();
            if (!nas.isEmpty()) {
                nas.replace('\\', '/');
                path.replace(nas, "$NASLOC");
            }
        }
#endif
        return path;
    }

    const qreal g_ctrlHeight = 24;

    QGraphicsItem* createItemWidget(
        const QVariant& value,
        zeno::ParamControl ctrl,
        zeno::ParamType type,
        CallbackCollection cbSet,
        QGraphicsScene* scene,
        const zeno::ControlProperty& controlProps
    )
    {
        ZtfUtil& inst = ZtfUtil::GetInstance();
        static NodeUtilParam m_nodeParams = inst.toUtilParam(inst.loadZtf(":/templates/node-example.xml"));
        QGraphicsItem* pItemWidget = nullptr;

        switch (ctrl)
        {
            case zeno::Lineedit:
            {
                const QString text = UiHelper::variantToString(value);

                ZEditableTextItem* pLineEdit = new ZEditableTextItem(text);
                pLineEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(64, zenoui::g_ctrlHeight)));
                pLineEdit->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
                pLineEdit->setNumSlider(scene, UiHelper::getSlideStep("", type));

                QObject::connect(pLineEdit, &ZEditableTextItem::editingFinished, [=]() {
                    // be careful about the dynamic type.
                    const QString textVal = pLineEdit->toPlainText();
                    const QVariant& newValue = UiHelper::parseStringByType(textVal, type);
                    if (newValue.type() == QVariant::String && type != zeno::Param_String)
                    {
                        if (!textVal.startsWith("="))
                            zeno::log_error("The formula '{}' need start with '='", textVal.toStdString());
                    }
                    cbSet.cbEditFinished(newValue);
                });
                pItemWidget = pLineEdit;
                break;
            }
            case zeno::Checkbox:
            {
                ZenoParamCheckBox* pCheckbox = new ZenoParamCheckBox;
                pCheckbox->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(20, 20)));
                pCheckbox->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
                pCheckbox->setData(GVKEY_TYPE, type);
                pCheckbox->setData(GVKEY_ALIGNMENT, Qt::AlignLeft);

                bool isChecked = value.toBool();
                pCheckbox->setCheckState(isChecked ? Qt::Checked : Qt::Unchecked);

                QObject::connect(pCheckbox, &ZenoParamCheckBox::stateChanged, [=](int state) {
                    bool bChecked = (state == Qt::Checked);
                    cbSet.cbEditFinished(bChecked);
                });
                pItemWidget = pCheckbox;
                break;
            }
            case zeno::ReadPathEdit:
            case zeno::WritePathEdit:
            case zeno::DirectoryPathEdit:
            {
                const QString& path = UiHelper::variantToString(value);
                ZenoParamPathEdit* pPathEditor = new ZenoParamPathEdit(path, ctrl, m_nodeParams.lineEditParam, cbSet.cbGetZsgDir);
                pPathEditor->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(200, zenoui::g_ctrlHeight)));
                pPathEditor->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pPathEditor->setData(GVKEY_TYPE, type);
                
                QObject::connect(pPathEditor, &ZenoParamPathEdit::editingFinished, [=]() {
                    cbSet.cbEditFinished(pPathEditor->text());
                });
                pItemWidget = pPathEditor;
                break;
            }
            case zeno::Multiline:
            {
                ZenoParamMultilineStr* pMultiStrEdit = new ZenoParamMultilineStr(
                    UiHelper::variantToString(value),
                    m_nodeParams.lineEditParam);

                pMultiStrEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 227))); //the height is the actual init size, hardcode it...
                pMultiStrEdit->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed));
                pMultiStrEdit->setData(GVKEY_TYPE, type);

                QObject::connect(pMultiStrEdit, &ZenoParamMultilineStr::editingFinished, [=]() {
                    const QString& newValue = pMultiStrEdit->text();
                    cbSet.cbEditFinished(newValue);
                });
                pItemWidget = pMultiStrEdit;
                break;
            }
            //case CONTROL_PYTHON_EDITOR:
            //{
            //    ZPythonEditorItem* pythonEditor = new ZPythonEditorItem(UiHelper::variantToString(value), m_nodeParams.lineEditParam);
            //    pythonEditor->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 227))); //the height is the actual init size, hardcode it...
            //    pythonEditor->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed));
            //    pythonEditor->setData(GVKEY_TYPE, type);

            //    QObject::connect(pythonEditor, &ZPythonEditorItem::editingFinished, [=]() {
            //        const QString& newValue = pythonEditor->text();
            //        cbSet.cbEditFinished(newValue);
            //    });
            //    pItemWidget = pythonEditor;
            //    break;
            //}
            case zeno::Heatmap:
            {
                ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);

                pEditBtn->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pEditBtn->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pEditBtn->setData(GVKEY_TYPE, type);

                QObject::connect(pEditBtn, &ZenoParamPushButton::clicked, [=]() {
                    QString val = cbSet.cbGetIndexData().toString();
                    ZenoHeatMapEditor editor(val);
                    editor.exec();
                    QString newVal = editor.colorRamps();
                    cbSet.cbEditFinished(QVariant::fromValue(newVal));
                    });
                pItemWidget = pEditBtn;
                break;
            }
            case zeno::PushButton:
            {
                //todo: name customize
                ZenoParamPushButton* pButton = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
                pButton->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pButton->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pButton->setData(GVKEY_TYPE, type);
                pItemWidget = pButton;
                QObject::connect(pButton, &ZenoParamPushButton::clicked, [=]() {
                    if (cbSet.cbBtnOnClicked)
                        cbSet.cbBtnOnClicked();
                });
                break;
            }
            case zeno::ColorVec:
            {
                QColor currentColor;
                if (ctrl == zeno::ColorVec) {
                    auto colorVec = value.value<UI_VECTYPE>();
                    colorVec.resize(3);
                    currentColor = QColor::fromRgbF(colorVec[0], colorVec[1], colorVec[2]);
                }
                
                ZenoParamPushButton *pEditBtn = new ZenoParamPushButton("", -1, QSizePolicy::Expanding);
                pEditBtn->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pEditBtn->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pEditBtn->setData(GVKEY_TYPE, type);
                pEditBtn->setProperty("color", currentColor.name());

                QObject::connect(pEditBtn, &ZenoParamPushButton::clicked, [=]() {
                    QColor color = ColorEditor::getColor(QColor(pEditBtn->property("color").toString()));
                    if (color.isValid()) 
                    {
                        pEditBtn->setProperty("color", color.name());
                        if (ctrl == zeno::ColorVec) {
                            UI_VECTYPE colorVec(3);
                            color.getRgbF(&colorVec[0], &colorVec[1], &colorVec[2]);
                            cbSet.cbEditFinished(QVariant::fromValue<UI_VECTYPE>(colorVec));
                        }
                    }
                });
                pItemWidget = pEditBtn;
                break;
            }
            case zeno::Vec2edit:
            case zeno::Vec3edit:
            case zeno::Vec4edit:
            {
                UI_VECTYPE vec = value.value<UI_VECTYPE>();

                int dim = -1;
                bool bFloat = false;
                if (ctrl == zeno::Vec2edit)
                {
                    dim = 2;
                    bFloat = type == zeno::Param_Vec2f;
                }
                else if (ctrl == zeno::Vec3edit)
                {
                    dim = 3;
                    bFloat = type == zeno::Param_Vec3f;
                }
                else if (ctrl == zeno::Vec4edit)
                {
                    dim = 4;
                    bFloat = type == zeno::Param_Vec4f;
                }
                if (vec.size() != dim)
                {
                    vec.resize(dim);
                }

                ZVecEditorItem* pVecEditor = new ZVecEditorItem(QVariant::fromValue(vec), bFloat, m_nodeParams.lineEditParam, scene);
                pVecEditor->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, zenoui::g_ctrlHeight)));
                pVecEditor->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed));
                pVecEditor->setData(GVKEY_TYPE, type);

                QObject::connect(pVecEditor, &ZVecEditorItem::editingFinished, [=]() {
                    const QVariant &newValue = pVecEditor->vec();
                    cbSet.cbEditFinished(newValue);
                });
                pItemWidget = pVecEditor;
                break;
            }
            case zeno::Combobox:
            {
                //todo: legacy case compatible
                QStringList items;
                if (controlProps.items.has_value())
                {
                    for (auto item : controlProps.items.value())
                        items.push_back(QString::fromStdString(item));
                }

                ZenoParamComboBox* pComboBox = new ZenoParamComboBox(items, m_nodeParams.comboboxParam);
                pComboBox->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pComboBox->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pComboBox->setData(GVKEY_TYPE, type);

                QString val = value.toString();
                if (items.indexOf(val) != -1)
                {
                    pComboBox->setText(val);
                }
                QObject::connect(pComboBox, &ZenoParamComboBox::textActivated, [=](const QString& textValue) {
                    QString oldValue = pComboBox->text();
                    cbSet.cbEditFinished(textValue);
                });
                pItemWidget = pComboBox;
                break;
            }
            case zeno::CurveEditor:
            {
                ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
                pEditBtn->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pEditBtn->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pEditBtn->setData(GVKEY_TYPE, type);
                QObject::connect(pEditBtn, &ZenoParamPushButton::clicked, [=]() {
                    ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);

                    QObject::connect(pEditor, &ZCurveMapEditor::finished, [=](int result) {
                        QString newVal = JsonHelper::dumpCurves(pEditor->curves());
                        cbSet.cbEditFinished(newVal);
                    });

                    pEditor->setAttribute(Qt::WA_DeleteOnClose);

                    const QString& str = cbSet.cbGetIndexData().toString();
                    CURVES_DATA curves = JsonHelper::parseCurves(str);
                    pEditor->addCurves(curves);
                    pEditor->exec();
                });
                pItemWidget = pEditBtn;
                break;
            }
            case zeno::Slider:
            {
                SLIDER_INFO sliderInfo;
                
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }

                ZenoParamSlider *pSlider = new ZenoParamSlider(Qt::Horizontal, value.toInt(), sliderInfo);
                pSlider->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, zenoui::g_ctrlHeight)));
                pSlider->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                QObject::connect(pSlider, &ZenoParamSlider::valueChanged, [=](int value) { 
                    cbSet.cbEditFinished(value);
                });
                pItemWidget = pSlider;
                break;
            }
            case zeno::SpinBox: 
            {
                SLIDER_INFO sliderInfo;
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }

                ZenoParamSpinBox *pSpinBox = new ZenoParamSpinBox(sliderInfo);
                pSpinBox->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pSpinBox->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pSpinBox->setValue(value.toInt());
                QObject::connect(pSpinBox, &ZenoParamSpinBox::valueChanged, [=](int value) { 
                    cbSet.cbEditFinished(value); 
                });
                pItemWidget = pSpinBox;
                break;
            }
            case zeno::SpinBoxSlider:
            {
                SLIDER_INFO sliderInfo;
                
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }

                ZenoParamSpinBoxSlider *pSlider = new ZenoParamSpinBoxSlider(Qt::Horizontal, value.toInt(), sliderInfo);
                pSlider->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, zenoui::g_ctrlHeight)));
                pSlider->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pSlider->setValue(value.toInt());
                QObject::connect(pSlider, &ZenoParamSpinBoxSlider::valueChanged, [=](int value) {
                    cbSet.cbEditFinished(value); 
                });
                pItemWidget = pSlider;
                break;
            }
            case zeno::DoubleSpinBox:
            {
                SLIDER_INFO sliderInfo;
                if (controlProps.ranges.has_value()) {
                    const auto& ranges = controlProps.ranges.value();
                    sliderInfo.min = ranges[0];
                    sliderInfo.max = ranges[1];
                    sliderInfo.step = ranges[2];
                }
                ZenoParamDoubleSpinBox* pSpinBox = new ZenoParamDoubleSpinBox(sliderInfo);
                pSpinBox->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pSpinBox->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pSpinBox->setValue(value.toDouble());
                QObject::connect(pSpinBox, &ZenoParamDoubleSpinBox::valueChanged, [=](double value) {
                    cbSet.cbEditFinished(value);
                });
                pItemWidget = pSpinBox;
                break;
            }
            default:
            {
                return nullptr;
            }
        }
        if (pItemWidget)
        {
            pItemWidget->setData(GVKEY_CONTROL, ctrl);
        }
        return pItemWidget;
    }
}

