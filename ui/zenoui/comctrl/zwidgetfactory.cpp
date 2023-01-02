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
#include "zspinboxslider.h"
#include "zdicttableview.h"


namespace zenoui
{
    QWidget* createWidget(
        const QVariant& value,
        PARAM_CONTROL ctrl,
        const QString& type,
        CallbackCollection cbSet,
        const QVariant& properties
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
                    cbSet.cbEditFinished(newValue);
                    });
                return pLineEdit;
            }
            case CONTROL_BOOL:
            {
                QCheckBox* pCheckbox = new QCheckBox;
                pCheckbox->setCheckState(value.toBool() ? Qt::Checked : Qt::Unchecked);
                QObject::connect(pCheckbox, &QCheckBox::stateChanged, [=](int state) {
                    cbSet.cbEditFinished(state);
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
                    cbSet.cbSwitch(true);
                    if (bRead) {
                        path = QFileDialog::getOpenFileName(nullptr, "File to Open", "", "All Files(*);;");
                    }
                    else {
                        path = QFileDialog::getSaveFileName(nullptr, "Path to Save", "", "All Files(*);;");
                    }
                    if (path.isEmpty())
                    {
                        cbSet.cbSwitch(false);
                        return;
                    }
                    pathLineEdit->setText(path);
                    emit pathLineEdit->textEditFinished();
                    pathLineEdit->clearFocus();
                    cbSet.cbSwitch(false);
                });
                QObject::connect(pathLineEdit, &ZLineEdit::textEditFinished, [=]() {
                    cbSet.cbEditFinished(pathLineEdit->text());
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
                pTextEdit->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);

                QTextCharFormat format;
                QFont font("HarmonyOS Sans", 12);
                format.setFont(font);
                pTextEdit->setCurrentFont(font);
                pTextEdit->setText(value.toString());

                QPalette pal = pTextEdit->palette();
                pal.setColor(QPalette::Base, QColor(37, 37, 37));
                pTextEdit->setPalette(pal);

                QObject::connect(pTextEdit, &ZTextEdit::editFinished, [=]() {
                    cbSet.cbEditFinished(pTextEdit->toPlainText());
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
                    cbSet.cbEditFinished(QVariant::fromValue(newGrad));
                });
                return pBtn;
            }
            case CONTROL_COLOR_NORMAL: {
                QPushButton *pBtn = new QPushButton;
                QPalette palette = pBtn->palette();
                pBtn->setFixedSize(ZenoStyle::dpiScaled(100), ZenoStyle::dpiScaled(30));
                palette.setColor(QPalette::Window, value.value<QColor>());
                pBtn->setStyleSheet(QString("background-color:%1; border:0;").arg(value.value<QColor>().name()));
                QObject::connect(pBtn, &QPushButton::clicked, [=]() {
                    QColor color = QColorDialog::getColor(pBtn->palette().window().color());
                    pBtn->setStyleSheet(QString("background-color:%1; border:0;").arg(color.name()));
                    cbSet.cbEditFinished(QVariant::fromValue(color));
                });
                return pBtn;
            }
            case CONTROL_VEC2_FLOAT:
            case CONTROL_VEC2_INT:
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

                ZVecEditor* pVecEdit = new ZVecEditor(vec, bFloat, dim, "zeno2_2_lineedit");
                QObject::connect(pVecEdit, &ZVecEditor::editingFinished, [=]() {
                    UI_VECTYPE vec = pVecEdit->vec();
                    const QVariant& newValue = QVariant::fromValue(vec);
                    cbSet.cbEditFinished(newValue);
                });
                return pVecEdit;
            }
            case CONTROL_ENUM:
            {
                QStringList items = properties.toStringList();
                QComboBox* pComboBox = new QComboBox;
                pComboBox->setProperty("cssClass", "newstyle");
                pComboBox->addItems(items);
                pComboBox->setCurrentText(value.toString());
                pComboBox->setEditable(true);
                pComboBox->setItemDelegate(new ZComboBoxItemDelegate2(pComboBox));

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
                QObject::connect(pComboBox, &QComboBox::textActivated, [=](const QString& text) {
                    cbSet.cbEditFinished(text);
                });
#else
                /*QObject::connect(pComboBox, &QComboBox::activated, [=](const QString& text) {
                    cbSet.cbEditFinished(text);
                });*/
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
                        cbSet.cbEditFinished(newValue);
                    });
                });
                return pBtn;
            }
            case CONTROL_HSLIDER:
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
                        background: #191D21;\
                    }\
                    \
                    QSlider::sub-page:horizontal {\
                        background: #707D9C;\
                    }\
                "));
                pSlider->setValue(value.toInt());

                SLIDER_INFO sliderInfo = properties.value<SLIDER_INFO>();
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
            case CONTROL_HSPINBOX:
            {
                QSpinBox* pSpinBox = new QSpinBox;
                pSpinBox->setProperty("cssClass", "control");
                pSpinBox->setAlignment(Qt::AlignCenter);
                return pSpinBox;
            }
            case CONTROL_SPINBOX_SLIDER:
            {
                ZSpinBoxSlider* pSlider = new ZSpinBoxSlider;
                SLIDER_INFO sliderInfo = properties.value<SLIDER_INFO>();
                pSlider->setSingleStep(sliderInfo.step);
                pSlider->setRange(sliderInfo.min, sliderInfo.max);
                pSlider->setValue(value.toInt());
                QObject::connect(pSlider, &ZSpinBoxSlider::valueChanged, [=](int value) {
                    cbSet.cbEditFinished(value);
                });
                return pSlider;
            }
            case CONTROL_DICTPANEL:
            {
                QAbstractItemModel* pModel = QVariantPtr<QAbstractItemModel>::asPtr(value);
                if (pModel)
                {
                    ZDictTableView* tableView = new ZDictTableView;
                    tableView->setModel(pModel);
                    QItemSelectionModel* pSelModel = tableView->selectionModel();
                    QObject::connect(pSelModel, &QItemSelectionModel::selectionChanged, [=](const QItemSelection& selected, const QItemSelection& deselected) {
                        auto lst = selected.indexes();
                        if (lst.size() == 1)
                        {
                            QModelIndex selIdx = lst[0];
                            if (selIdx.column() == 1)
                            {
                                QModelIndex linkIdx = selIdx.data(ROLE_LINK_IDX).toModelIndex();
                                QModelIndex outNodeIdx = linkIdx.data(ROLE_OUTNODE_IDX).toModelIndex();
                                if (cbSet.cbNodeSelected)
                                    cbSet.cbNodeSelected(outNodeIdx);
                            }
                        }
                    });
                    return tableView;
                }
                break;
            }
            default:
                return nullptr;
        }
        return nullptr;
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
        case CONTROL_VEC2_FLOAT:
        case CONTROL_VEC2_INT:
        case CONTROL_VEC3_FLOAT:
        case CONTROL_VEC3_INT:
        case CONTROL_VEC4_FLOAT:
        case CONTROL_VEC4_INT:return qobject_cast<ZVecEditor*>(pControl) != nullptr;
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