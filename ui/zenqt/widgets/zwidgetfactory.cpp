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
#include <zeno/utils/helper.h>


using namespace zeno::reflect;

namespace zenoui
{
    QWidget* createWidget(
        const QModelIndex& nodeIdx,
        const Any& value,                   //编辑值，里面的类型未必是参数真正的类型，比如公式的编辑值
        zeno::ParamControl ctrl,
        const zeno::ParamType paramType,    //参数真正的类型
        CallbackCollection cbSet,
        const Any& controlProps
    )
    {
        switch (ctrl)
        {
            case zeno::Lineedit:
            {
                if (paramType == gParamType_String) {
                    QString text = QString::fromStdString(any_cast<std::string>(value));
                    ZLineEdit* pLineEdit = new ZLineEdit(text);
                    pLineEdit->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                    pLineEdit->setProperty("cssClass", "zeno2_2_lineedit");
                    pLineEdit->setNodeIdx(nodeIdx);
                    QObject::connect(pLineEdit, &ZLineEdit::editingFinished, [=]() {
                        const std::string& newValue = pLineEdit->text().toStdString();
                        cbSet.cbEditFinished(newValue);
                    });
                    return pLineEdit;
                }
                else {
                    ZASSERT_EXIT(value.type().hash_code() == gParamType_PrimVariant, nullptr);
                    const zeno::PrimVar& var = any_cast<zeno::PrimVar>(value);

                    ZCoreParamLineEdit* pLineEdit = new ZCoreParamLineEdit(var, paramType);

                    pLineEdit->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                    pLineEdit->setProperty("cssClass", "zeno2_2_lineedit");
                    pLineEdit->setNumSlider(UiHelper::getSlideStep("", paramType));
                    pLineEdit->setNodeIdx(nodeIdx);
                    QObject::connect(pLineEdit, &ZCoreParamLineEdit::valueChanged, [=](zeno::PrimVar newVal) {
                        cbSet.cbEditFinished(newVal);
                    });
                    return pLineEdit;
                }
            }
            case zeno::Checkbox:
            {
                QCheckBox* pCheckbox = new QCheckBox;
                size_t typecode = value.type().hash_code();
                if (typecode == gParamType_Bool) {
                    pCheckbox->setCheckState(any_cast<bool>(value) ? Qt::Checked : Qt::Unchecked);
                }
                else if (typecode == gParamType_Int) {
                    //以后反射定义里不排除用户会写int类型却用checkbox
                    pCheckbox->setCheckState(any_cast<int>(value) ? Qt::Checked : Qt::Unchecked);
                }
                else {
                    ZASSERT_EXIT(false, nullptr);
                }

                QObject::connect(pCheckbox, &QCheckBox::stateChanged, [=](int state) {
                    bool bChecked = state == Qt::Checked;
                    cbSet.cbEditFinished(bChecked);
                });
                return pCheckbox;
            }
            case zeno::ReadPathEdit:
            case zeno::WritePathEdit:
            case zeno::DirectoryPathEdit:
            {
                if (value.type().hash_code() != gParamType_String) {
                    ZASSERT_EXIT(false, nullptr);
                }
                QString text = QString::fromStdString(any_cast<std::string>(value));
                ZPathEdit *pathLineEdit = new ZPathEdit(text, ctrl);
                pathLineEdit->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                pathLineEdit->setProperty("control", ctrl);
                
                QObject::connect(pathLineEdit, &ZLineEdit::textEditFinished, [=]() {
                    const std::string& path = pathLineEdit->text().toStdString();
                    cbSet.cbEditFinished(path);
                });
                return pathLineEdit;
            }
            case zeno::Multiline:
            {
                if (value.type().hash_code() != gParamType_String) {
                    ZASSERT_EXIT(false, nullptr);
                }
                QString text = QString::fromStdString(any_cast<std::string>(value));

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
                pTextEdit->setText(text);

                QPalette pal = pTextEdit->palette();
                pal.setColor(QPalette::Base, QColor(61, 61, 61));
                pTextEdit->setPalette(pal);

                QObject::connect(pTextEdit, &ZTextEdit::editFinished, [=]() {
                    const std::string& editedText = pTextEdit->toPlainText().toStdString();
                    cbSet.cbEditFinished(editedText);
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
                    //TODO
                    /*
                    QString val = cbSet.cbGetIndexData().toString();
                    ZenoHeatMapEditor editor(val);
                    editor.exec();
                    QString newVal = editor.colorRamps();
                    cbSet.cbEditFinished(QVariant::fromValue(newVal));
                    */
                });
                return pBtn;
            }
            case zeno::ColorVec:
            {
                if (value.type().hash_code() != gParamType_Vec3f) {
                    ZASSERT_EXIT(false, nullptr);
                }

                zeno::vec3f colorVec = any_cast<zeno::vec3f>(value);
                QColor currentColor = QColor::fromRgbF(colorVec[0], colorVec[1], colorVec[2]);

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
                            zeno::vec3f newVal(colorVec[0], colorVec[1], colorVec[2]);
                            cbSet.cbEditFinished(newVal);
                        }
                    }
                });
                return pBtn;
            }
            case zeno::Vec2edit:
            case zeno::Vec3edit:
            case zeno::Vec4edit:
            {
                bool bFloat = false;
                if (paramType == zeno::types::gParamType_Vec2i || paramType == zeno::types::gParamType_Vec2f)
                {
                    bFloat = paramType == zeno::types::gParamType_Vec2f;
                }
                else if (paramType == zeno::types::gParamType_Vec3i || paramType == zeno::types::gParamType_Vec3f)
                {
                    bFloat = paramType == zeno::types::gParamType_Vec3f;
                }
                else if (paramType == zeno::types::gParamType_Vec4i || paramType == zeno::types::gParamType_Vec4f)
                {
                    bFloat = paramType == zeno::types::gParamType_Vec4f;
                }
                else {
                    ZASSERT_EXIT(false, nullptr);
                }

                ZASSERT_EXIT(value.type().hash_code() == gParamType_VecEdit, nullptr);

                ZVecEditor* pVecEdit = new ZVecEditor(any_cast<zeno::vecvar>(value), bFloat, "zeno2_2_lineedit");
                pVecEdit->setNodeIdx(nodeIdx);
                pVecEdit->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                QObject::connect(pVecEdit, &ZVecEditor::valueChanged, [=](zeno::vecvar newValue) {
                    cbSet.cbEditFinished(newValue);
                });
                return pVecEdit;
            }
            case zeno::Combobox:
            {
                ZASSERT_EXIT(paramType == gParamType_String, nullptr);
                QString text = QString::fromStdString(any_cast<std::string>(value));

                QStringList items;
                if (controlProps.has_value())
                {
                    auto& vec = zeno::reflect::any_cast<std::vector<std::string>>(controlProps);
                    for (auto item : vec)
                        items.push_back(QString::fromStdString(item));
                }

                ZComboBox *pComboBox = new ZComboBox;
                pComboBox->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                pComboBox->addItems(items);
                pComboBox->setCurrentText(text);
                pComboBox->setEditable(true);
                pComboBox->setItemDelegate(new ZComboBoxItemDelegate2(pComboBox));

                QObject::connect(pComboBox, &ZComboBox::_textActivated, [=](const QString& newText) {
                    cbSet.cbEditFinished(newText.toStdString());
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
                        zeno::CurvesData& newVal = pEditor->curves();
                        auto& anyVal = zeno::reflect::make_any<zeno::CurvesData>(newVal);
                        cbSet.cbEditFinished(anyVal);
                    });

                    zeno::CurvesData curves;
                    if (cbSet.cbGetIndexData)
                    {
                        auto& qvar = cbSet.cbGetIndexData();
                        if (qvar.canConvert<zeno::reflect::Any>()) {
                            const auto& anyVal = qvar.value<zeno::reflect::Any>();
                            if (zeno::reflect::get_type<zeno::CurvesData>() == anyVal.type()) {
                                curves = zeno::reflect::any_cast<zeno::CurvesData>(anyVal);
                            }
                        }
                    }
                    pEditor->addCurves(curves);
                    pEditor->exec();
                });
                return pBtn;
            }
            case zeno::Slider:
            {
                ZASSERT_EXIT(paramType == gParamType_Int, nullptr);
                int intVal = any_cast<int>(value);

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
                pSlider->setValue(intVal);

                SLIDER_INFO sliderInfo;
                
                if (controlProps.has_value()) {
                    auto& vec = zeno::reflect::any_cast<std::vector<float>>(controlProps);
                    ZASSERT_EXIT(vec.size() == 3, nullptr);
                    sliderInfo.min = vec[0];
                    sliderInfo.max = vec[1];
                    sliderInfo.step = vec[2];
                }
                pSlider->setSingleStep(sliderInfo.step);
                pSlider->setRange(sliderInfo.min, sliderInfo.max);

                QObject::connect(pSlider, &QSlider::valueChanged, [=](int newVal) {
                    cbSet.cbEditFinished(newVal);
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
                ZASSERT_EXIT(paramType == gParamType_Int, nullptr);
                int intVal = any_cast<int>(value);

                QSpinBox* pSpinBox = new QSpinBox;
                pSpinBox->setProperty("cssClass", "control");
                pSpinBox->setAlignment(Qt::AlignCenter);
                pSpinBox->setValue(intVal);
                pSpinBox->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                SLIDER_INFO sliderInfo;
                
                if (controlProps.has_value()) {
                    auto& vec = zeno::reflect::any_cast<std::vector<float>>(controlProps);
                    ZASSERT_EXIT(vec.size() == 3, nullptr);
                    sliderInfo.min = vec[0];
                    sliderInfo.max = vec[1];
                    sliderInfo.step = vec[2];
                }
                pSpinBox->setSingleStep(sliderInfo.step);
                pSpinBox->setRange(sliderInfo.min, sliderInfo.max);
                QObject::connect(pSpinBox, static_cast<void (QSpinBox::*)(int)>(& QSpinBox::valueChanged),[=](int newVal) { 
                    cbSet.cbEditFinished(newVal);
                });
                return pSpinBox;
            }
            case zeno::DoubleSpinBox: {
                ZASSERT_EXIT(paramType == gParamType_Float, nullptr);
                float fVal = any_cast<float>(value);

                QDoubleSpinBox *pSpinBox = new QDoubleSpinBox;
                pSpinBox->setProperty("cssClass", "control");
                pSpinBox->setAlignment(Qt::AlignCenter);
                pSpinBox->setValue(fVal);
                pSpinBox->setFixedHeight(ZenoStyle::dpiScaled(zenoui::g_ctrlHeight));
                SLIDER_INFO sliderInfo;
                if (controlProps.has_value()) {
                    auto& vec = zeno::reflect::any_cast<std::vector<float>>(controlProps);
                    ZASSERT_EXIT(vec.size() == 3, nullptr);
                    sliderInfo.min = vec[0];
                    sliderInfo.max = vec[1];
                    sliderInfo.step = vec[2];
                }
                pSpinBox->setSingleStep(sliderInfo.step);
                pSpinBox->setRange(sliderInfo.min, sliderInfo.max);
                QObject::connect(pSpinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),[=](double newVal) {
                    cbSet.cbEditFinished(newVal);
                });
                return pSpinBox;
            }
            case zeno::SpinBoxSlider:
            {
                ZASSERT_EXIT(paramType == gParamType_Int, nullptr);
                int intVal = any_cast<int>(value);

                ZSpinBoxSlider* pSlider = new ZSpinBoxSlider;
                SLIDER_INFO sliderInfo;
                if (controlProps.has_value()) {
                    auto& vec = zeno::reflect::any_cast<std::vector<float>>(controlProps);
                    ZASSERT_EXIT(vec.size() == 3, nullptr);
                    sliderInfo.min = vec[0];
                    sliderInfo.max = vec[1];
                    sliderInfo.step = vec[2];
                }
                pSlider->setSingleStep(sliderInfo.step);
                pSlider->setRange(sliderInfo.min, sliderInfo.max);
                pSlider->setValue(intVal);
                QObject::connect(pSlider, &ZSpinBoxSlider::valueChanged, [=](int newVal) {
                    cbSet.cbEditFinished(newVal);
                });
                return pSlider;
            }
            case zeno::CodeEditor:
            {
                ZASSERT_EXIT(paramType == gParamType_String, nullptr);
                QString text = QString::fromStdString(any_cast<std::string>(value));

                ZCodeEditor* pCodeEditor = new ZCodeEditor(text);
                QObject::connect(pCodeEditor, &ZCodeEditor::editFinished, [=](const QString& newText) {
                    cbSet.cbEditFinished(newText.toStdString());
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
}