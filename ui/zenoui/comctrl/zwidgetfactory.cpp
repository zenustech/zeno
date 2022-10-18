#include "zwidgetfactory.h"
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/zlinewidget.h>
#include <zenoui/comctrl/zlineedit.h>
#include <zenoui/comctrl/ztextedit.h>
#include <zenoui/comctrl/dialog/curvemap/zcurvemapeditor.h>
#include <zenoui/comctrl/dialog/zenoheatmapeditor.h>
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/curvemodel.h>
#include <zenoui/comctrl/zveceditor.h>
#include "view/zcomboboxitemdelegate.h"
#include "variantptr.h"
#include "zassert.h"


namespace zenoui
{
    QWidget* createWidget(
        const QVariant& value,
        PARAM_CONTROL ctrl,
        const QString& type,
        Callback_EditFinished cbFunc,
        CALLBACK_SWITCH cbSwitch,
        const QVariant& supply
    )
    {
        switch (ctrl)
        {
            case CONTROL_INT:
            case CONTROL_FLOAT:
            case CONTROL_STRING:
            {
                ZLineEdit* pLineEdit = new ZLineEdit(UiHelper::variantToString(value));
                pLineEdit->setProperty("cssClass", "zeno2_2_lineedit");
                pLineEdit->setNumSlider(UiHelper::getSlideStep("", ctrl));
                QObject::connect(pLineEdit, &ZLineEdit::editingFinished, [=]() {
                    // be careful about the dynamic type.
                    const QVariant& newValue = UiHelper::parseStringByType(pLineEdit->text(), type);
                    cbFunc(newValue);
                    });
                return pLineEdit;
            }
            case CONTROL_BOOL:
            {
                QCheckBox* pCheckbox = new QCheckBox;
                pCheckbox->setStyleSheet("\
                    QCheckBox::indicator{\
                        width: 16px;\
                        height: 16px;\
                    }\
                    QCheckBox::indicator:unchecked {\
                        image: url(:/icons/checkbox-idle.svg);\
                    }\
                    QCheckBox::indicator:checked {\
                        image: url(:/icons/checkbox-light.svg);\
                    }\
                ");
                pCheckbox->setCheckState(value.toBool() ? Qt::Checked : Qt::Unchecked);
                QObject::connect(pCheckbox, &QCheckBox::stateChanged, [=](int state) {
                    cbFunc(state);
                });
                return pCheckbox;
            }
            case CONTROL_READPATH:
            case CONTROL_WRITEPATH:
            {
                ZLineEdit* pathLineEdit = new ZLineEdit(value.toString());
                pathLineEdit->setIcons(":/icons/file-loader.svg", ":/icons/file-loader-on.svg");
                pathLineEdit->setProperty("cssClass", "zeno2_2_lineedit");
                pathLineEdit->setProperty("control", ctrl);
                pathLineEdit->setFocusPolicy(Qt::ClickFocus);
                QObject::connect(pathLineEdit, &ZLineEdit::btnClicked, [=]() {
                    bool bRead = ctrl == CONTROL_READPATH;
                    QString path;
                    cbSwitch(true);
                    if (bRead) {
                        path = QFileDialog::getOpenFileName(nullptr, "File to Open", "", "All Files(*);;");
                    }
                    else {
                        path = QFileDialog::getSaveFileName(nullptr, "Path to Save", "", "All Files(*);;");
                    }
                    if (path.isEmpty())
                    {
                        cbSwitch(false);
                        return;
                    }
                    pathLineEdit->setText(path);
                    emit pathLineEdit->textEditFinished();
                    pathLineEdit->clearFocus();
                    cbSwitch(false);
                });
                QObject::connect(pathLineEdit, &ZLineEdit::textEditFinished, [=]() {
                    cbFunc(pathLineEdit->text());
                });
                return pathLineEdit;
            }
            case CONTROL_MULTILINE_STRING:
            {
                ZTextEdit* pTextEdit = new ZTextEdit;
                pTextEdit->setFrameShape(QFrame::NoFrame);
                pTextEdit->setProperty("cssClass", "proppanel");
                pTextEdit->setProperty("control", ctrl);
                pTextEdit->setFont(QFont("HarmonyOS Sans", 12));

                QTextCharFormat format;
                QFont font("HarmonyOS Sans", 12);
                format.setFont(font);
                pTextEdit->setCurrentFont(font);
                pTextEdit->setText(value.toString());

                QPalette pal = pTextEdit->palette();
                pal.setColor(QPalette::Base, QColor(37, 37, 37));
                pTextEdit->setPalette(pal);

                QObject::connect(pTextEdit, &ZTextEdit::editFinished, [=]() {
                    cbFunc(pTextEdit->toPlainText());
                });
                return pTextEdit;
            }
            case CONTROL_COLOR:
            {
                QPushButton* pBtn = new QPushButton("Edit Heatmap");
                pBtn->setProperty("cssClass", "proppanel");
                QObject::connect(pBtn, &QPushButton::clicked, [=]() {
                    QLinearGradient grad = value.value<QLinearGradient>();
                    ZenoHeatMapEditor editor(grad);
                    editor.exec();
                    QLinearGradient newGrad = editor.colorRamps();
                    cbFunc(QVariant::fromValue(newGrad));
                });
                return pBtn;
            }
            case CONTROL_DICTKEY:
            {
                return nullptr; //no control expect key editor.
            }
            case CONTROL_VEC:
            {
                UI_VECTYPE vec = value.value<UI_VECTYPE>();
                int dim = -1;
                bool bFloat = false;
                UiHelper::parseVecType(type, dim, bFloat);

                ZVecEditor* pVecEdit = new ZVecEditor(vec, bFloat, 3, "zeno2_2_lineedit");
                QObject::connect(pVecEdit, &ZVecEditor::editingFinished, [=]() {
                    UI_VECTYPE vec = pVecEdit->vec();
                    const QVariant& newValue = QVariant::fromValue(vec);
                    cbFunc(newValue);
                });
                return pVecEdit;
            }
            case CONTROL_ENUM:
            {
                //QStringList items = type.mid(QString("enum ").length()).split(QRegExp("\\s+"));
                QComboBox* pComboBox = new QComboBox;

                const QStringList& items = supply.toStringList();
                pComboBox->setProperty("cssClass", "newstyle");
                pComboBox->addItems(items);
                pComboBox->setCurrentText(value.toString());
                pComboBox->setEditable(true);
                pComboBox->setItemDelegate(new ZComboBoxItemDelegate2(pComboBox));

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
                QObject::connect(pComboBox, &QComboBox::textActivated, [=](const QString& text) {
                    cbFunc(text);
                });
#else
                QObject::connect(pComboBox, &QComboBox::activated, [=](const QString& text) {
                    cbFunc(text);
                });
#endif
                return pComboBox;
            }
            case CONTROL_CURVE:
            {
                QPushButton* pBtn = new QPushButton("Edit Curve");
                pBtn->setProperty("cssClass", "proppanel");
                QObject::connect(pBtn, &QPushButton::clicked, [=]() {
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
                return pBtn;
            }
            default:
                return nullptr;
        }
    }

    bool isMatchControl(PARAM_CONTROL ctrl, QWidget* pControl)
    {
        if (!pControl)
            return false;

        switch (ctrl)
        {
        case CONTROL_STRING:
        case CONTROL_INT:
        case CONTROL_FLOAT:    return qobject_cast<ZLineEdit*>(pControl) != nullptr;    //be careful type changed.
        case CONTROL_READPATH:
        case CONTROL_WRITEPATH: return qobject_cast<ZLineEdit*>(pControl) != nullptr;
        case CONTROL_BOOL:    return qobject_cast<QCheckBox*>(pControl) != nullptr;
        case CONTROL_VEC:    return qobject_cast<ZVecEditor*>(pControl) != nullptr;
        case CONTROL_ENUM:    return qobject_cast<QComboBox*>(pControl) != nullptr;
        case CONTROL_MULTILINE_STRING:    return qobject_cast<ZTextEdit*>(pControl) != nullptr;
        case CONTROL_CURVE:
        case CONTROL_COLOR:    return qobject_cast<QPushButton*>(pControl) != nullptr;
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
            pVecEditor->setVec(value.value<UI_VECTYPE>(), pVecEditor->isFloat());
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