#include "framework.h"
#include "zpropertiespanel.h"
#include "common.h"
#include "designermainwin.h"
#include "styletabwidget.h"
#include "nodesview.h"
#include "nodeswidget.h"


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

ImageGroupBox::ImageGroupBox(QWidget* parent)
    : QGroupBox(parent)
{
    setTitle("Image");

    QVBoxLayout *pGbImageLayout = new QVBoxLayout;

    QHBoxLayout *pNormalImage = new QHBoxLayout;
    pNormalImage->addWidget(new QLabel("Normal:"));
    m_pNormal = new QLabel("");
    pNormalImage->addWidget(m_pNormal);
    QPushButton *pBtn1 = new QPushButton("...");
    pNormalImage->addWidget(pBtn1);

    QHBoxLayout *pHovered = new QHBoxLayout;
    pHovered->addWidget(new QLabel("Hovered:"));
    m_pHovered = new QLabel("");
    pHovered->addWidget(m_pHovered);
    QPushButton *pBtn2 = new QPushButton("...");
    pHovered->addWidget(pBtn2);

    QHBoxLayout *pSelected = new QHBoxLayout;
    pSelected->addWidget(new QLabel("Selected:"));
    m_pSelected = new QLabel("");
    pSelected->addWidget(m_pSelected);
    QPushButton *pBtn3 = new QPushButton("...");
    pSelected->addWidget(pBtn3);

    pGbImageLayout->addLayout(pNormalImage);
    pGbImageLayout->addLayout(pHovered);
    pGbImageLayout->addLayout(pSelected);

    connect(pBtn1, &QPushButton::clicked, this, [=]() {
        QString original = QFileDialog::getOpenFileName(this, tr("Select an image"), ".", "Svg Files (*.svg)\nJPEG (*.jpg *jpeg)\nGIF (*.gif)\nPNG (*.png)\nBitmap Files (*.bmp))");
        if (original.isEmpty())
            return;
        QFileInfo f(original);
        QString fn = f.fileName();
        m_pNormal->setText(fn);
        m_normal = original;
        emit normalImported(m_normal);
    });

    connect(pBtn2, &QPushButton::clicked, this, [=]() {
        QString original = QFileDialog::getOpenFileName(this, tr("Select an image"), ".", "Svg Files (*.svg)\nJPEG (*.jpg *jpeg)\nGIF (*.gif)\nPNG (*.png)\nBitmap Files (*.bmp))");
        if (original.isEmpty())
            return;
        QFileInfo f(original);
        QString fn = f.fileName();
        m_pHovered->setText(fn);
        m_hovered = original;
        emit hoverImported(m_hovered);
    });

    connect(pBtn3, &QPushButton::clicked, this, [=]() {
        QString original = QFileDialog::getOpenFileName(this, tr("Select an image"), ".", "Svg Files (*.svg)\nJPEG (*.jpg *jpeg)\nGIF (*.gif)\nPNG (*.png)\nBitmap Files (*.bmp))");
        if (original.isEmpty())
            return;
        QFileInfo f(original);
        QString fn = f.fileName();
        m_pSelected->setText(fn);
        m_selected = original;
        emit selectedImported(m_selected);
    });

    setLayout(pGbImageLayout);
}

ColorWidget::ColorWidget(QWidget *parent)
    : QWidget(parent)
{
}

QSize ColorWidget::sizeHint() const
{
    return QSize(18, 18);
}

void ColorWidget::mouseReleaseEvent(QMouseEvent* event)
{
    QColorDialog dlg(this);
    dlg.setWindowFlag(Qt::SubWindow);
    dlg.setWindowTitle(tr("More Color"));
    if (dlg.exec() == QDialog::Accepted)
    {
        QColor clr = dlg.selectedColor();
        if (clr.isValid())
        {
            m_color = clr;
            emit colorChanged(m_color);
        }
    }
}

void ColorWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    QRect rc = rect();
    painter.fillRect(rc, m_color);
}


TextGroupBox::TextGroupBox(QWidget *parent)
    : QGroupBox(parent), m_fontsize(12)
{
    setTitle("Text");

    QVBoxLayout *pLayout = new QVBoxLayout;

    QGridLayout* pGridLayout = new QGridLayout;
    QFontComboBox* pComboBox = new QFontComboBox;
    pGridLayout->addWidget(new QLabel("Font:"),0,0);
    pGridLayout->addWidget(pComboBox,0,1);

    pGridLayout->addWidget(new QLabel("Font Size:"),1,0);
    QSpinBox *spinBox = new QSpinBox;
    auto lineEdit = spinBox->findChild<QLineEdit *>();
    if (lineEdit) {
        lineEdit->setReadOnly(true);
        lineEdit->setFocusPolicy(Qt::NoFocus);
    }
    spinBox->setMinimum(5);
    spinBox->setMaximum(32);
    spinBox->setValue(m_fontsize);
    pGridLayout->addWidget(spinBox,1,1);

    pGridLayout->addWidget(new QLabel("Text:"), 2, 0);
    QLineEdit* pLineEdit = new QLineEdit;
    pGridLayout->addWidget(pLineEdit, 2, 1);

    QHBoxLayout* pHLayout2 = new QHBoxLayout;
    pHLayout2->addWidget(new QLabel("Color:"));
    m_colorWidget = new ColorWidget;
    pHLayout2->addWidget(m_colorWidget);
    pHLayout2->addStretch();

    pLayout->addLayout(pGridLayout);
    pLayout->addLayout(pHLayout2);

    m_font = pComboBox->font();
    m_font.setPointSize(m_fontsize);

    setLayout(pLayout);

    connect(spinBox, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(pLineEdit, &QLineEdit::editingFinished, this, [=]() {
        QString text = pLineEdit->text();
        emit textChanged(text);
    });
    connect(pComboBox, &QFontComboBox::currentFontChanged, this, [=](const QFont &f) {
        m_font = f;
        emit fontChanged(m_font, m_color);
    });

    connect(m_colorWidget, &ColorWidget::colorChanged, this, [=](QColor color) {
        m_color = color;
        emit fontChanged(m_font, m_color);
    });
}

void TextGroupBox::onValueChanged(int value)
{
    m_fontsize = value;
    m_font.setPointSize(m_fontsize);
    emit fontChanged(m_font, m_color);
}

TransformGroupBox::TransformGroupBox(QWidget* parent)
    : QGroupBox(parent)
{
    setTitle("Transform");

    QGridLayout *pLayout = new QGridLayout;

    m_pX = new ValueInputWidget("X:");
    m_pY = new ValueInputWidget("Y:");
    m_pWidth = new ValueInputWidget("W:");
    m_pHeight = new ValueInputWidget("H:");

    pLayout->addWidget(m_pX, 0, 0);
    pLayout->addWidget(m_pWidth, 0, 1);
    pLayout->addWidget(m_pY, 1, 0);
    pLayout->addWidget(m_pHeight, 1, 1);

    connect(m_pX, SIGNAL(valueChanged()), this, SIGNAL(valueChanged()));
    connect(m_pY, SIGNAL(valueChanged()), this, SIGNAL(valueChanged()));
    connect(m_pWidth, SIGNAL(valueChanged()), this, SIGNAL(valueChanged()));
    connect(m_pHeight, SIGNAL(valueChanged()), this, SIGNAL(valueChanged()));

    setLayout(pLayout);
}

void TransformGroupBox::setValue(const qreal& x, const qreal& y, const qreal& w, const qreal& h)
{
    m_pX->setValue(x);
    m_pY->setValue(y);
    m_pWidth->setValue(w);
    m_pHeight->setValue(h);
}

bool TransformGroupBox::getValue(qreal& x, qreal& y, qreal& w, qreal& h)
{
    bool bOk = false;
    x = m_pX->value(bOk);
    if (!bOk)
        return false;
    y = m_pY->value(bOk);
    if (!bOk)
        return false;
    w = m_pWidth->value(bOk);
    if (!bOk)
        return false;
    h = m_pHeight->value(bOk);
    if (!bOk)
        return false;
    return true;
}


ZComponentPropPanel::ZComponentPropPanel(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pVBoxLayout = new QVBoxLayout;

    m_pGbTransform = new TransformGroupBox;
    pVBoxLayout->addWidget(m_pGbTransform);

    m_pGbImage = new ImageGroupBox;
    pVBoxLayout->addWidget(m_pGbImage);

    m_pGbText = new TextGroupBox;
    pVBoxLayout->addWidget(m_pGbText);

    pVBoxLayout->addStretch();

    m_pGbText->setVisible(false);
    m_pGbImage->setVisible(false);
    m_pGbTransform->setVisible(false);

    setLayout(pVBoxLayout);
}

void ZComponentPropPanel::initModel()
{
    DesignerMainWin* pWin = getMainWindow(this);
    if (auto tab = pWin->getCurrentTab())
    {
        setEnabled(true);
        QStandardItemModel* model = tab->model();
        QItemSelectionModel* selection= tab->selectionModel();
        connect(model, &QStandardItemModel::itemChanged, this, [=](QStandardItem *pItem) {
            QModelIndexList lst = selection->selectedIndexes();
            if (lst.empty())
                return;
            QRectF rc = lst[0].data(NODEPOS_ROLE).toRectF();
            m_pGbTransform->setValue(rc.left(), rc.top(), rc.width(), rc.height());
        });

        connect(selection, &QItemSelectionModel::selectionChanged, this, &ZComponentPropPanel::onSelectionChanged);
        bool ret = connect(m_pGbTransform, &TransformGroupBox::valueChanged, this, [=] {
            onUpdateModel(model, selection);
            });
        ret = connect(m_pGbImage, &ImageGroupBox::normalImported, this, [=](QString imgPath) {
                QModelIndexList lst = selection->selectedIndexes();
                if (lst.empty())
                    return;
                QStandardItem *pItem = model->itemFromIndex(lst[0]);
                pItem->setData(imgPath, NODEPATH_ROLE);
            });
        ret = connect(m_pGbImage, &ImageGroupBox::hoverImported, this, [=](QString imgPath) {
                QModelIndexList lst = selection->selectedIndexes();
                if (lst.empty())
                    return;
                QStandardItem *pItem = model->itemFromIndex(lst[0]);
                pItem->setData(imgPath, NODEHOVERPATH_ROLE);
            });
        ret = connect(m_pGbImage, &ImageGroupBox::selectedImported, this, [=](QString imgPath) {
                QModelIndexList lst = selection->selectedIndexes();
                if (lst.empty())
                    return;
                QStandardItem *pItem = model->itemFromIndex(lst[0]);
                pItem->setData(imgPath, NODESELECTEDPATH_ROLE);
            });
        
        //text
        connect(m_pGbText, &TextGroupBox::fontChanged, this, [=](QFont font, QColor color) {
            QModelIndexList lst = selection->selectedIndexes();
            if (lst.empty())
                return;
            QStandardItem *pItem = model->itemFromIndex(lst[0]);
            pItem->setData(font, NODEFONT_ROLE);
            pItem->setData(color, NODEFONTCOLOR_ROLE);
        });
        connect(m_pGbText, &TextGroupBox::textChanged, this, [=](QString text) {
            QModelIndexList lst = selection->selectedIndexes();
            if (lst.empty())
                return;
            QStandardItem *pItem = model->itemFromIndex(lst[0]);
            pItem->setData(text, NODETEXT_ROLE);
        });
    }
    else
    {
        setEnabled(false);
    }
}

void ZComponentPropPanel::onUpdateModel(QStandardItemModel* model, QItemSelectionModel* selection)
{
    QModelIndexList lst = selection->selectedIndexes();
    if (lst.empty()) return;

    QModelIndex index = lst[0];
    QString id = index.data(NODEID_ROLE).toString();
    QStandardItem* pItem = model->itemFromIndex(index);
    if (pItem)
    {
        qreal x, y, w, h;
        bool bOk = m_pGbTransform->getValue(x, y, w, h);
        if (bOk)
            pItem->setData(QRectF(x, y, w, h), NODEPOS_ROLE);
    }
}

void ZComponentPropPanel::onModelDataChanged(QStandardItem* pItem)
{
    QRectF rc = pItem->data(NODEPOS_ROLE).toRectF();
    m_pGbTransform->setValue(rc.left(), rc.top(), rc.width(), rc.height());
}

void ZComponentPropPanel::onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected)
{
    QModelIndexList lst = selected.indexes();
    bool hasSelection = !lst.isEmpty();
    m_pGbTransform->setEnabled(hasSelection);
    m_pGbTransform->setVisible(true);
    if (hasSelection)
    {
        QModelIndex idx = lst.at(0);
        QString id = idx.data(NODEID_ROLE).toString();
        QRectF rc = idx.data(NODEPOS_ROLE).toRectF();
        m_pGbTransform->setValue(rc.left(), rc.top(), rc.width(), rc.height());

        NODE_CONTENT content = (NODE_CONTENT) idx.data(NODECONTENT_ROLE).toInt();
        m_pGbImage->setVisible(content == NC_IMAGE);
        m_pGbText->setVisible(content == NC_TEXT);
    }
    else {
        m_pGbImage->setVisible(false);
        m_pGbText->setVisible(false);
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