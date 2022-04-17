#include "framework.h"
#include "nodeswidget.h"
#include "nodesview.h"
#include "nodescene.h"
#include <render/ztfutil.h>
#include <comctrl/zobjectbutton.h>


NodesWidget::NodesWidget(QWidget *parent)
    : QWidget(parent)
    , m_pView(nullptr)
    , m_factor(1.0)
    , m_bShowBdr(true)
{
    init(":/templates/node-empty.xml");
    m_fileName = "node";
}

NodesWidget::NodesWidget(const QString &filePath, QWidget *parent)
    : QWidget(parent)
    , m_pView(nullptr)
    , m_factor(1.0)
    , m_filePath(filePath)
{
    QFileInfo fileInfo(m_filePath);
    m_fileName = fileInfo.fileName();
    init(filePath);
}

void NodesWidget::init(const QString& filePath)
{
    QVBoxLayout *pLayout = new QVBoxLayout;

    QHBoxLayout *pHLayout = new QHBoxLayout;
    ZMiniToolButton *pUndo = new ZMiniToolButton(QIcon(":/icons/undo.png"));
    ZMiniToolButton *pRedo = new ZMiniToolButton(QIcon(":/icons/redo.png"));
    ZMiniToolButton *pSnapGrid = new ZMiniToolButton(QIcon(":/icons/magnet_grid.png"));
    pSnapGrid->setCheckable(true);
    ZMiniToolButton *pSnapPoint = new ZMiniToolButton(QIcon(":/icons/magnet_point.png"));
    pSnapPoint->setCheckable(true);
    ZMiniToolButton *pShowBorder = new ZMiniToolButton(QIcon(":/icons/showborder.png"));
    pShowBorder->setCheckable(true);
    pShowBorder->setChecked(true);

    pHLayout->addWidget(pUndo);
    pHLayout->addWidget(pRedo);
    pHLayout->addWidget(pSnapGrid);
    pHLayout->addWidget(pSnapPoint);
    pHLayout->addWidget(pShowBorder);
    pHLayout->addStretch();
    pLayout->addLayout(pHLayout);

    m_pView = new NodesView(this);
    m_pView->initSkin(filePath);
    m_pView->initNode();

    pLayout->addWidget(m_pView);
    setLayout(pLayout);

    connect(pSnapGrid, &ZMiniToolButton::clicked, [=]() {
        if (pSnapGrid->isChecked()) {
            m_snap = SNAP_GRID;
        } else {
            m_snap = NO_SNAP;
        }
        pSnapPoint->setChecked(false);
    });

    connect(pSnapPoint, &ZMiniToolButton::clicked, [=]() {
        if (pSnapPoint->isChecked()) {
            m_snap = SNAP_PIXEL;
        } else {
            m_snap = NO_SNAP;
        }
        pSnapGrid->setChecked(false);
    });

    connect(pShowBorder, &ZMiniToolButton::clicked, [=]() {
        m_bShowBdr = pShowBorder->isChecked();
        m_pView->scene()->update();
    });

    new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_S), this, SLOT(save()));
    new QShortcut(QKeySequence(Qt::SHIFT + Qt::Key_O), this, SLOT(resetPreset()));
}

void NodesWidget::setFactor(const qreal &factor)
{
    m_factor = factor;
}

qreal NodesWidget::factor() const
{
    return m_factor;
}

QStandardItemModel* NodesWidget::model() const
{
    return m_pView->scene()->model();
}

QItemSelectionModel* NodesWidget::selectionModel() const
{
    return m_pView->scene()->selectionModel();
}

void NodesWidget::resetPreset()
{
    QDialog dlg(this);

    QVBoxLayout *pLayout = new QVBoxLayout;

    QGridLayout *pGridLayout = new QGridLayout;
    QLineEdit *pWidth = new QLineEdit;
    QLineEdit *pHeight = new QLineEdit;
    pGridLayout->addWidget(new QLabel("Width:"), 0, 0);
    pGridLayout->addWidget(pWidth, 0, 1);
    pGridLayout->addWidget(new QLabel("Height:"), 1, 0);
    pGridLayout->addWidget(pHeight, 1, 1);
    pLayout->addLayout(pGridLayout);

    QHBoxLayout *pHLayout = new QHBoxLayout;
    QPushButton *pOK = new QPushButton("OK");
    QPushButton *pCancel = new QPushButton("Cancel");
    pHLayout->addStretch();
    pHLayout->addWidget(pOK);
    pHLayout->addWidget(pCancel);
    pLayout->addLayout(pHLayout);
    bool bRet = connect(pOK, SIGNAL(clicked()), &dlg, SLOT(accept()));
    connect(pCancel, SIGNAL(clicked()), &dlg, SLOT(reject()));

    dlg.setLayout(pLayout);
    if (dlg.exec() == QDialog::Accepted)
    {
        bool bOK = false;
        int W = pWidth->text().toInt(&bOK);
        if (!bOK) return;
        int H = pHeight->text().toInt(&bOK);
        if (!bOK) return;

        m_pView->resetPreset(W, H);
    }
}

void NodesWidget::save()
{
    if (m_filePath.isEmpty())
    {
        const QString &initialPath = ".";
        QFileDialog fileDialog(this, tr("Save As"), initialPath);
        fileDialog.setAcceptMode(QFileDialog::AcceptSave);
        fileDialog.setFileMode(QFileDialog::AnyFile);
        fileDialog.setDirectory(initialPath);
        if (fileDialog.exec() != QDialog::Accepted)
            return;

        m_filePath = fileDialog.selectedFiles().first();
        QFileInfo fileInfo(m_filePath);
        m_fileName = fileInfo.fileName();
    }
    ZtfUtil::GetInstance().exportZtf(m_pView->scene()->exportNodeParam(), m_filePath);
    markDirty(false);
}

void NodesWidget::markDirty(bool dirty)
{
    m_dirty = dirty;
    emit tabDirtyChanged(m_dirty);
}

QString NodesWidget::fileName() const
{
    return m_fileName;
}