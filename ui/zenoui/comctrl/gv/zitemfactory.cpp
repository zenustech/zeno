#include "zitemfactory.h"
#include <zenomodel/include/uihelper.h>
#include <zenoui/render/ztfutil.h>
#include <zenoui/comctrl/gv/zlineedititem.h>
#include <zenomodel/include/curvemodel.h>
#include "zveceditoritem.h"
#include "style/zenostyle.h"
#include "../dialog/zenoheatmapeditor.h"
#include "../dialog/curvemap/zcurvemapeditor.h"
#include "variantptr.h"
#include "zassert.h"
#include "zgraphicstextitem.h"


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



    QValidator* validateForSockets(PARAM_CONTROL ctrl)
    {
        switch (ctrl)
        {
        case CONTROL_INT:       return new QIntValidator;
        case CONTROL_FLOAT:     return new QDoubleValidator;
        }
        return nullptr;
    }


    QGraphicsItem* createItemWidget(
        const QVariant& value,
        PARAM_CONTROL ctrl,
        const QString& type,
        Callback_EditFinished cbFunc,
        QGraphicsScene* scene,
        CALLBACK_SWITCH cbSwitch,
        const QVariant& controlProps
    )
    {
        ZtfUtil& inst = ZtfUtil::GetInstance();
        static NodeUtilParam m_nodeParams = inst.toUtilParam(inst.loadZtf(":/templates/node-example.xml"));
        QGraphicsItem* pItemWidget = nullptr;

        switch (ctrl)
        {
            case CONTROL_INT:
            case CONTROL_FLOAT:
            case CONTROL_STRING:
            {
                const QString text = UiHelper::variantToString(value);
#ifdef ENABLE_WIDGET_LINEEDIT
                ZenoParamLineEdit *pLineEdit = new ZenoParamLineEdit(text, ctrl, m_nodeParams.lineEditParam);
                pLineEdit->setValidator(validateForSockets(ctrl));
                pLineEdit->setNumSlider(scene, UiHelper::getSlideStep("", ctrl));
                pLineEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 32)));
                pLineEdit->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed));
                pLineEdit->setData(GVKEY_TYPE, type);
                QObject::connect(pLineEdit, &ZenoParamLineEdit::editingFinished, [=]() {
                    // be careful about the dynamic type.
                    const QVariant& newValue = UiHelper::parseStringByType(pLineEdit->text(), type);
                    cbFunc(newValue);
                    });
                pItemWidget = pLineEdit;
#else
                ZEditableTextItem* pLineEdit = new ZEditableTextItem(text);
                pLineEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 28)));
                pLineEdit->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pLineEdit->setNumSlider(scene, UiHelper::getSlideStep("", ctrl));

                if (ctrl == CONTROL_INT)
                {
                    pLineEdit->setValidator(new QIntValidator(pLineEdit));
                }

                QObject::connect(pLineEdit, &ZEditableTextItem::editingFinished, [=]() {
                    // be careful about the dynamic type.
                    const QString textVal = pLineEdit->toPlainText();
                    const QVariant& newValue = UiHelper::parseStringByType(textVal, type);
                    cbFunc(newValue);
                });
                pItemWidget = pLineEdit;
#endif
                break;
            }
            case CONTROL_BOOL:
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
                    cbFunc(bChecked);
                });
                pItemWidget = pCheckbox;
                break;
            }
            case CONTROL_READPATH:
            case CONTROL_WRITEPATH:
            {
                const QString& path = UiHelper::variantToString(value);
                ZenoParamPathEdit* pPathEditor = new ZenoParamPathEdit(path, ctrl, m_nodeParams.lineEditParam);

                pPathEditor->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 32)));
                pPathEditor->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pPathEditor->setData(GVKEY_TYPE, type);

                bool isRead = ctrl == CONTROL_READPATH;

                QObject::connect(pPathEditor, &ZenoParamPathEdit::clicked, [=]() {
                    cbSwitch(true);
                    QString path;
                    if (isRead) {
                        path = getOpenFileName(ZenoParamPathEdit::tr("File to Open"), "", ZenoParamPathEdit::tr("All Files(*);;"));
                    }
                    else {
                        path = getSaveFileName(ZenoParamPathEdit::tr("Path to Save"), "", ZenoParamPathEdit::tr("All Files(*);;"));
                    }
                    if (path.isEmpty())
                    {
                        cbSwitch(false);
                        return;
                    }
                    pPathEditor->setPath(path);
                    cbSwitch(false);
                });
                QObject::connect(pPathEditor, &ZenoParamPathEdit::pathValueChanged, [=](QString newPath) {
                    cbFunc(newPath);
                    });
                pItemWidget = pPathEditor;
                break;
            }
            case CONTROL_MULTILINE_STRING:
            {
                ZenoParamMultilineStr* pMultiStrEdit = new ZenoParamMultilineStr(
                    UiHelper::variantToString(value),
                    m_nodeParams.lineEditParam);

                pMultiStrEdit->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 227))); //the height is the actual init size, hardcode it...
                pMultiStrEdit->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed));
                pMultiStrEdit->setData(GVKEY_TYPE, type);

                QObject::connect(pMultiStrEdit, &ZenoParamMultilineStr::editingFinished, [=]() {
                    const QString& newValue = pMultiStrEdit->text();
                    cbFunc(newValue);
                });
                pItemWidget = pMultiStrEdit;
                break;
            }
            case CONTROL_COLOR:
            {
                QLinearGradient grad = value.value<QLinearGradient>();
                ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);

                pEditBtn->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 32)));
                pEditBtn->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pEditBtn->setData(GVKEY_TYPE, type);

                QObject::connect(pEditBtn, &ZenoParamPushButton::clicked, [=]() {
                    ZenoHeatMapEditor editor(grad);
                    editor.exec();
                    QLinearGradient newGrad = editor.colorRamps();
                    cbFunc(QVariant::fromValue(newGrad));
                    });
                pItemWidget = pEditBtn;
                break;
            }
            case CONTROL_VEC2_INT:
            case CONTROL_VEC2_FLOAT:
            case CONTROL_VEC3_FLOAT:
            case CONTROL_VEC3_INT:
            case CONTROL_VEC4_FLOAT:
            case CONTROL_VEC4_INT:
            {
                UI_VECTYPE vec = value.value<UI_VECTYPE>();

                int dim = -1;
                bool bFloat = false;
                if (ctrl == CONTROL_VEC2_INT || ctrl == CONTROL_VEC2_FLOAT)
                {
                    dim = 2;
                    bFloat = ctrl == CONTROL_VEC2_FLOAT;
                }
                else if (ctrl == CONTROL_VEC3_INT || ctrl == CONTROL_VEC3_FLOAT)
                {
                    dim = 3;
                    bFloat = ctrl == CONTROL_VEC3_FLOAT;
                }
                else if (ctrl == CONTROL_VEC4_INT || ctrl == CONTROL_VEC4_FLOAT)
                {
                    dim = 4;
                    bFloat = ctrl == CONTROL_VEC4_FLOAT;
                }

                if (vec.size() != dim)
                {
                    vec.resize(dim);
                }

                ZVecEditorItem* pVecEditor = new ZVecEditorItem(vec, bFloat, m_nodeParams.lineEditParam, scene);
                pVecEditor->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 32)));
                pVecEditor->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed));
                pVecEditor->setData(GVKEY_TYPE, type);

                QObject::connect(pVecEditor, &ZVecEditorItem::editingFinished, [=]() {
                    UI_VECTYPE vec = pVecEditor->vec();
                    const QVariant& newValue = QVariant::fromValue(vec);
                    cbFunc(newValue);
                });
                pItemWidget = pVecEditor;
                break;
            }
            case CONTROL_ENUM:
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
                pComboBox->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 32)));
                pComboBox->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pComboBox->setData(GVKEY_TYPE, type);

                QString val = value.toString();
                if (items.indexOf(val) != -1)
                {
                    pComboBox->setText(val);
                }
                QObject::connect(pComboBox, &ZenoParamComboBox::textActivated, [=](const QString& textValue) {
                    QString oldValue = pComboBox->text();
                    cbFunc(textValue);
                });
                pItemWidget = pComboBox;
                break;
            }
            case CONTROL_CURVE:
            {
                ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
                pEditBtn->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, 32)));
                pEditBtn->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
                pEditBtn->setData(GVKEY_TYPE, type);
                QObject::connect(pEditBtn, &ZenoParamPushButton::clicked, [=]() {
                    ZCurveMapEditor* pEditor = new ZCurveMapEditor(true);
                    pEditor->setAttribute(Qt::WA_DeleteOnClose);
                    // what if value changed? removed?
                    CurveModel* pModel = QVariantPtr<CurveModel>::asPtr(value);
                    ZASSERT_EXIT(pModel);
                    pEditor->addCurve(pModel);
                    pEditor->show();

                    QObject::connect(pEditor, &ZCurveMapEditor::finished, [=](int result) {
                        ZASSERT_EXIT(pEditor->curveCount() == 1);
                        CurveModel* pCurveModel = pEditor->getCurve(0);
                        const QVariant& newValue = QVariantPtr<CurveModel>::asVariant(pCurveModel);
                        cbFunc(newValue);
                        });
                    });
                pItemWidget = pEditBtn;
                break;
            }
            case CONTROL_HSLIDER: 
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
                QObject::connect(pSlider, &ZenoParamSlider::valueChanged, [=](int value) { 
					cbFunc(value);
				});
                pItemWidget = pSlider;
                break;
			}
            case CONTROL_HSPINBOX: 
			{
                ZenoParamSpinBox *pSpinBox = new ZenoParamSpinBox;
                pSpinBox->setValue(value.toInt());
                QObject::connect(pSpinBox, &ZenoParamSpinBox::valueChanged, [=](int value) { 
					cbFunc(value); 
				});
                pItemWidget = pSpinBox;
                break;
            }
            case CONTROL_SPINBOX_SLIDER: 
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
                pSlider->setValue(value.toInt());
                QObject::connect(pSlider, &ZenoParamSpinBoxSlider::valueChanged, [=](int value) {
					cbFunc(value); 
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

