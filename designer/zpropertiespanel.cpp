#include "framework.h"
#include "zpropertiespanel.h"
#include "common.h"
#include "designermainwin.h"
#include "styletabwidget.h"
#include "nodesview.h"


DesignerMainWin* getMainWindow(QWidget* pWidget)
{
    QWidget* p = pWidget;
    while (p)
    {
        if (DesignerMainWin* pWin = qobject_cast<DesignerMainWin*>(p))
        {
            return pWin;
        }
        p = p->parentWidget();
    }
    return nullptr;
}


ValueInputWidget::ValueInputWidget(const QString& name, QWidget* parent)
    : QWidget(parent)
    , m_pSpinBox(nullptr)
    , m_pLineEdit(nullptr)
{
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->addWidget(new QLabel(name));

    m_pLineEdit = new QLineEdit;
    pLayout->addWidget(m_pLineEdit);

    connect(m_pLineEdit, SIGNAL(returnPressed()), this, SIGNAL(valueChanged()));

    setLayout(pLayout);
}

void ValueInputWidget::setValue(qreal value)
{
    m_pLineEdit->setText(QString::number(value));
}

qreal ValueInputWidget::value(bool& bOk)
{
    float value = m_pLineEdit->text().toFloat(&bOk);
    return value;
}

ZPagePropPanel::ZPagePropPanel(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pVBoxLayout = new QVBoxLayout;

    pVBoxLayout->addWidget(new QLabel("Grid Settings"));

    QLabel* pLabel1 = new QLabel("Transform");
    pVBoxLayout->addWidget(pLabel1);

    QHBoxLayout* pHLayout = new QHBoxLayout;
    m_pWidth = new ValueInputWidget("W:");
    m_pHeight = new ValueInputWidget("H:");
    pHLayout->addWidget(m_pWidth);
    pHLayout->addWidget(m_pHeight);
    pVBoxLayout->addLayout(pHLayout);

    QFrame* pLine = new QFrame;
    pLine->setFrameShape(QFrame::HLine);
    pVBoxLayout->addWidget(pLine);

    pVBoxLayout->addWidget(new QLabel("Color"));
    pVBoxLayout->addStretch();

    setLayout(pVBoxLayout);
}

ZComponentPropPanel::ZComponentPropPanel(QWidget* parent)
    : QWidget(parent)
    , m_pX(new ValueInputWidget("X:"))
    , m_pY(new ValueInputWidget("Y:"))
    , m_pWidth(new ValueInputWidget("W:"))
    , m_pHeight(new ValueInputWidget("H:"))
{
    QVBoxLayout* pVBoxLayout = new QVBoxLayout;

    pVBoxLayout->addWidget(new QLabel("Transform"));

    QGridLayout* pLayout = new QGridLayout;

    m_pX = new ValueInputWidget("X:");
    m_pY = new ValueInputWidget("Y:");
    m_pWidth = new ValueInputWidget("W:");
    m_pHeight = new ValueInputWidget("H:");

    pLayout->addWidget(m_pX, 0, 0);
    pLayout->addWidget(m_pWidth, 0, 1);
    pLayout->addWidget(m_pY, 1, 0);
    pLayout->addWidget(m_pHeight, 1, 1);

    pVBoxLayout->addLayout(pLayout);
    pVBoxLayout->addStretch();

    setLayout(pVBoxLayout);
}

void ZComponentPropPanel::initModel()
{
    DesignerMainWin* pWin = getMainWindow(this);
    if (auto view = pWin->getTabWidget()->getCurrentView())
    {
        QStandardItemModel* model = view->findChild<QStandardItemModel*>(NODE_MODEL_NAME);
        QItemSelectionModel* selection= view->findChild<QItemSelectionModel*>(NODE_SELECTION_MODEL);
        connect(model, SIGNAL(itemChanged(QStandardItem*)), this, SLOT(onModelDataChanged(QStandardItem*)));
        connect(selection, &QItemSelectionModel::selectionChanged, this, &ZComponentPropPanel::onSelectionChanged);
        bool ret = connect(m_pX, &ValueInputWidget::valueChanged, this, [=] {
            onUpdateModel(model, selection);
            });
        connect(m_pY, &ValueInputWidget::valueChanged, this, [=] {
            onUpdateModel(model, selection);
            });
        connect(m_pWidth, &ValueInputWidget::valueChanged, this, [=] {
            onUpdateModel(model, selection);
            });
        connect(m_pHeight, &ValueInputWidget::valueChanged, this, [=] {
            onUpdateModel(model, selection);
            });
    }
}

void ZComponentPropPanel::onUpdateModel(QStandardItemModel* model, QItemSelectionModel* selection)
{
    QModelIndex index = selection->currentIndex();
    QStandardItem* pItem = model->itemFromIndex(index);
    if (pItem)
    {
        bool bOk = false;
        qreal x = m_pX->value(bOk);
        if (!bOk) return;
        qreal y = m_pY->value(bOk);
        if (!bOk) return;
        qreal w = m_pWidth->value(bOk);
        if (!bOk) return;
        qreal h = m_pHeight->value(bOk);
        if (!bOk) return;

        pItem->setData(QRectF(x, y, w, h), NODEPOS_ROLE);
    }
}

void ZComponentPropPanel::onModelDataChanged(QStandardItem* pItem)
{
    QRectF rc = pItem->data(NODEPOS_ROLE).toRectF();
    m_pX->setValue(rc.left());
    m_pY->setValue(rc.top());
    m_pWidth->setValue(rc.width());
    m_pHeight->setValue(rc.height());
}

void ZComponentPropPanel::onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected)
{
    QModelIndexList lst = selected.indexes();
    bool hasSelection = !lst.isEmpty();

    m_pX->setEnabled(hasSelection); m_pY->setEnabled(hasSelection);
    m_pWidth->setEnabled(hasSelection); m_pHeight->setEnabled(hasSelection);

    if (hasSelection)
    {
        QModelIndex idx = lst.at(0);
        NODE_ID id = (NODE_ID)idx.data(Qt::UserRole + 1).toInt();
        
        QRectF rc = idx.data(NODEPOS_ROLE).toRectF();
        m_pX->setValue(rc.left());
        m_pY->setValue(rc.top());
        m_pWidth->setValue(rc.width());
        m_pHeight->setValue(rc.height());
    }
}


ZElementPropPanel::ZElementPropPanel(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pVBoxLayout = new QVBoxLayout;
    pVBoxLayout->addWidget(new QLabel("assets"));
    
    QHBoxLayout* pHLayout = new QHBoxLayout;
    m_pAsset = new QLabel("example.svg");
    pHLayout->addWidget(m_pAsset);

    QPushButton* pBtnFile = new QPushButton("...");
    pHLayout->addWidget(pBtnFile);

    pVBoxLayout->addLayout(pHLayout);

    QFrame* pLine = new QFrame;
    pLine->setFrameShape(QFrame::HLine);
    pVBoxLayout->addWidget(pLine);

    pVBoxLayout->addWidget(new QLabel("Transform"));

    QGridLayout* pLayout = new QGridLayout;

    m_pX = new ValueInputWidget("X:");
    m_pY = new ValueInputWidget("Y:");
    m_pWidth = new ValueInputWidget("W:");
    m_pHeight = new ValueInputWidget("H:");

    pLayout->addWidget(m_pX, 0, 0);
    pLayout->addWidget(m_pWidth, 0, 1);
    pLayout->addWidget(m_pY, 1, 0);
    pLayout->addWidget(m_pHeight, 1, 1);

    pVBoxLayout->addLayout(pLayout);

    pVBoxLayout->addStretch();
    
    setLayout(pVBoxLayout);
}