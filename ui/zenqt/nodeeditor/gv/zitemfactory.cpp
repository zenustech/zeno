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
        const QVariant& controlProps
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
                pLineEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pLineEdit->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
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
                    if (pLineEdit->showSlider())
                        cbSet.cbEditFinishedWithSlider(newValue);
                    else
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
            case zeno::Pathedit:
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
                QLinearGradient grad = value.value<QLinearGradient>();
                ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);

                pEditBtn->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(100, zenoui::g_ctrlHeight)));
                pEditBtn->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pEditBtn->setData(GVKEY_TYPE, type);

                QObject::connect(pEditBtn, &ZenoParamPushButton::clicked, [=]() {
                    ZenoHeatMapEditor editor(grad);
                    editor.exec();
                    QLinearGradient newGrad = editor.colorRamps();
                    cbSet.cbEditFinished(QVariant::fromValue(newGrad));
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
            case zeno::Color:
            case zeno::ColorVec:
            {
                QColor currentColor;
                if (ctrl == zeno::Color) {
                    currentColor = value.value<QColor>();
                }
                else if (ctrl == zeno::ColorVec) {
                    auto colorVec = value.value<UI_VECTYPE>();
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
                        if (ctrl == zeno::Color) {
                            cbSet.cbEditFinished(QVariant::fromValue(color));
                        }
                        else if (ctrl == zeno::ColorVec) {
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
                bool bFloat = type == zeno::Param_Float;
                if (ctrl == zeno::Vec2edit)
                {
                    dim = 2;
                }
                else if (ctrl == zeno::Vec3edit)
                {
                    dim = 3;
                }
                else if (ctrl == zeno::Vec4edit)
                {
                    dim = 4;
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
                    if (pVecEditor->hasSliderShow())
                        cbSet.cbEditFinishedWithSlider(newValue);
                    else
                        cbSet.cbEditFinished(newValue);
                });
                pItemWidget = pVecEditor;
                break;
            }
            case zeno::Combobox:
            {
                //todo: legacy case compatible
                QStringList items;
                if (controlProps.type() == QMetaType::QVariantMap)
                {
                    QVariantMap props = controlProps.toMap();
                    if (props.find("items") != props.end())
                        items = props["items"].toStringList();
                }
                else if (controlProps.type() == QVariant::StringList)
                {
                    items = controlProps.toStringList();
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
                        cbSet.cbEditFinished(QVariant::fromValue(pEditor->curves()));
                    });

                    pEditor->setAttribute(Qt::WA_DeleteOnClose);

                    CURVES_DATA curves = cbSet.cbGetIndexData().value<CURVES_DATA>();
                    pEditor->addCurves(curves);
                    pEditor->exec();
                });
                pItemWidget = pEditBtn;
                break;
            }
            case zeno::Slider:
            {
                SLIDER_INFO sliderInfo;
                if (controlProps.type() == QMetaType::QVariantMap) {
                    QVariantMap props = controlProps.toMap();
                    if (props.contains("min") && props.contains("max") && props.contains("step")) {
                        sliderInfo.min = props["min"].toInt();
                        sliderInfo.max = props["max"].toInt();
                        sliderInfo.step = props["step"].toInt();
                    }
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
                if (controlProps.type() == QMetaType::QVariantMap) {
                    QVariantMap props = controlProps.toMap();
                    if (props.contains("min") && props.contains("max") && props.contains("step")) {
                        sliderInfo.min = props["min"].toInt();
                        sliderInfo.max = props["max"].toInt();
                        sliderInfo.step = props["step"].toInt();
                    }
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
                if (controlProps.type() == QMetaType::QVariantMap) {
                    QVariantMap props = controlProps.toMap();
                    if (props.contains("min") && props.contains("max") && props.contains("step")) {
                        sliderInfo.min = props["min"].toInt();
                        sliderInfo.max = props["max"].toInt();
                        sliderInfo.step = props["step"].toInt();
                    }
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

