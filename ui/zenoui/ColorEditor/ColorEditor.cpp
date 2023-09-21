#include "ColorEditor.h"

#include <queue>

#include <QApplication>
#include <QCursor>
#include <QDebug>
#include <QDesktopWidget>
#include <QDialogButtonBox>
#include <QDrag>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QScreen>
#include <QScrollBar>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QVBoxLayout>

#include "zenoui/style/zenostyle.h"

//--------------------------------------------------------- color wheel ------------------------------------------------
class ColorWheel::Private
{
public:
    static constexpr int selectorRadius = 4;
    static constexpr int comboSelectorRadius = 3;
    int radius = 0;
    QColor selectedColor = QColor(Qt::white);
    QImage colorBuffer;
    colorcombo::ICombination* colorCombination = nullptr;

    void renderWheel(const QRect& rect)
    {
        auto center = rect.center();
        auto size = rect.size();

        radius = std::min(rect.width(), rect.height()) / 2 - selectorRadius;

        // init buffer
        colorBuffer = QImage(size, QImage::Format_ARGB32);
        colorBuffer.fill(Qt::transparent);

        // create gradient
        QConicalGradient hsvGradient(center, 0);
        for (int deg = 0; deg < 360; deg += 60) {
            hsvGradient.setColorAt(deg / 360.0, QColor::fromHsvF(deg / 360.0, 1.0, selectedColor.valueF()));
        }
        hsvGradient.setColorAt(1.0, QColor::fromHsvF(0.0, 1.0, selectedColor.valueF()));

        QRadialGradient valueGradient(center, radius);
        valueGradient.setColorAt(0.0, QColor::fromHsvF(0.0, 0.0, selectedColor.valueF()));
        valueGradient.setColorAt(1.0, Qt::transparent);

        QPainter painter(&colorBuffer);
        painter.setRenderHint(QPainter::Antialiasing, true);
        // draw color wheel
        painter.setPen(Qt::transparent);
        painter.setBrush(hsvGradient);
        painter.drawEllipse(center, radius, radius);
        painter.setBrush(valueGradient);
        painter.drawEllipse(center, radius, radius);
    }
};

ColorWheel::ColorWheel(QWidget* parent)
    : QWidget(parent)
    , p(new Private)
{
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ColorWheel::setColorCombination(colorcombo::ICombination* combination)
{
    p->colorCombination = combination;
    repaint();
}

void ColorWheel::setSelectedColor(const QColor& color)
{
    if (!isEnabled()) return;

    if (color.value() != p->selectedColor.value()) {
        p->selectedColor = color;
        p->renderWheel(this->rect());
    }
    else {
        p->selectedColor = color;
    }
    update();
}

QColor ColorWheel::getSelectedColor() const
{
    return p->selectedColor;
}

QColor ColorWheel::getColor(int x, int y) const
{
    if (p->radius <= 0) return QColor();

    auto line = QLineF(this->rect().center(), QPointF(x, y));
    auto h = line.angle() / 360.0;
    auto s = std::min(1.0, line.length() / p->radius);
    auto v = p->selectedColor.valueF();
    return QColor::fromHsvF(h, s, v);
}

void ColorWheel::paintEvent(QPaintEvent* e)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    // draw wheel
    painter.drawImage(0, 0, p->colorBuffer);
    // draw selected color circle
    painter.setPen(Qt::black);
    painter.setBrush(Qt::white);
    drawSelector(&painter, p->selectedColor, p->selectorRadius);
    // draw color combination circle
    if (p->colorCombination) {
        auto colors = p->colorCombination->genColors(p->selectedColor);
        for (const auto& color : colors) {
            drawSelector(&painter, color, p->comboSelectorRadius);
        }
        // add selected color, so the user can switch between this
        colors.push_back(p->selectedColor);
        emit combinationColorChanged(colors);
    }
}

void ColorWheel::mousePressEvent(QMouseEvent* e)
{
    processMouseEvent(e);
}

void ColorWheel::mouseMoveEvent(QMouseEvent* e)
{
    processMouseEvent(e);
}

void ColorWheel::resizeEvent(QResizeEvent* e)
{
    p->renderWheel(this->rect());
}

void ColorWheel::processMouseEvent(QMouseEvent* e)
{
    if (e->buttons() & Qt::LeftButton) {
        p->selectedColor = getColor(e->x(), e->y());
        emit colorSelected(p->selectedColor);
        update();
    }
}

void ColorWheel::drawSelector(QPainter* painter, const QColor& color, int radius)
{
    auto line = QLineF::fromPolar(color.hsvSaturationF() * p->radius, color.hsvHueF() * 360.0);
    line.translate(this->rect().center());
    painter->drawEllipse(line.p2(), radius, radius);
}

//-------------------------------------------------- color combination --------------------------------------------
namespace colorcombo
{
ICombination::ICombination(QObject* parent)
    : QObject(parent)
    , m_min(0)
    , m_max(1)
    , m_value(0)
    , m_rangeEnabled(false)
{
}

ICombination::ICombination(double min, double max, double value, bool rangeEnabled, QObject* parent)
    : QObject(parent)
    , m_min(min)
    , m_max(max)
    , m_value(value)
    , m_rangeEnabled(rangeEnabled)
{
}

QString ICombination::name()
{
    return tr("None");
}

QVector<QColor> ICombination::genColors(const QColor& color)
{
    return {};
}

void ICombination::setRange(double min, double max)
{
    m_min = min;
    m_max = max;
}

void ICombination::setValue(double value)
{
    m_value = value;
}

double ICombination::min() const
{
    return m_min;
}

double ICombination::max() const
{
    return m_max;
}

double ICombination::getValue() const
{
    return m_value;
}

bool ICombination::rangeEnabled() const
{
    return m_rangeEnabled;
}

Complementary::Complementary(QObject* parent)
    : ICombination(parent)
{
}

QString Complementary::name()
{
    return tr("Complementary");
}

QVector<QColor> Complementary::genColors(const QColor& color)
{
    return {QColor::fromHsv((color.hsvHue() + 180) % 360, color.hsvSaturation(), color.value())};
}

Monochromatic::Monochromatic(QObject* parent)
    : ICombination(0, 100, 50, true, parent)
{
}

QString Monochromatic::name()
{
    return tr("Monochromatic");
}

QVector<QColor> Monochromatic::genColors(const QColor& color)
{
    double rate = getValue() / (max() - min());
    return {QColor::fromHsvF(color.hsvHueF(), color.hsvSaturationF(), color.valueF() * rate)};
}

Analogous::Analogous(QObject* parent)
    : ICombination(0, 180, 30, true, parent)
{
}

QString Analogous::name()
{
    return tr("Analogous");
}

QVector<QColor> Analogous::genColors(const QColor& color)
{
    int add = getValue();
    return {QColor::fromHsv((color.hsvHue() + add) % 360, color.hsvSaturation(), color.value()),
            QColor::fromHsv((color.hsvHue() - add + 360) % 360, color.hsvSaturation(), color.value())};
}

Triadic::Triadic(QObject* parent)
    : ICombination(0, 180, 120, true, parent)
{
}

QString Triadic::name()
{
    return tr("Triadic");
}

QVector<QColor> Triadic::genColors(const QColor& color)
{
    int add = getValue();
    return {QColor::fromHsv((color.hsvHue() + add) % 360, color.hsvSaturation(), color.value()),
            QColor::fromHsv((color.hsvHue() - add + 360) % 360, color.hsvSaturation(), color.value())};
}

Tetradic::Tetradic(QObject* parent)
    : ICombination(-90, 90, 90, true, parent)
{
}

QString Tetradic::name()
{
    return tr("Tetradic");
}

QVector<QColor> Tetradic::genColors(const QColor& color)
{
    /*
     * A--------B
     * |        |
     * D--------C
     *
     * A : H, S, V
     * B : H - 90 + factor * 180, S, V
     * C : H + 180, S, V
     * D : H + 90 + factor * 180, S, V
     */
    int add = getValue();
    return {QColor::fromHsv((color.hsvHue() + add + 360) % 360, color.hsvSaturation(), color.value()),
            QColor::fromHsv((color.hsvHue() + 180) % 360, color.hsvSaturation(), color.value()),
            QColor::fromHsv((color.hsvHue() + add + 180 + 360) % 360, color.hsvSaturation(), color.value())};
}
} // namespace colorcombo

//--------------------------------------------------- color slider -------------------------------------------
void JumpableSlider::mousePressEvent(QMouseEvent* e)
{
    if (e->button() == Qt::LeftButton) {
        e->accept();
        setSliderDown(true);
        handleMouseEvent(e);
    }
    else {
        QSlider::mousePressEvent(e);
    }
}

void JumpableSlider::mouseMoveEvent(QMouseEvent* e)
{
    if (e->buttons() & Qt::LeftButton) {
        e->accept();
        handleMouseEvent(e);
    }
    else {
        QSlider::mouseMoveEvent(e);
    }
}

void JumpableSlider::mouseReleaseEvent(QMouseEvent* e)
{
    QSlider::mouseReleaseEvent(e);
}

void JumpableSlider::handleMouseEvent(QMouseEvent* e)
{
    int newVal;
    if (orientation() == Qt::Horizontal) {
        newVal = minimum() + ((maximum() - minimum() + 1) * e->x()) / width();
    }
    else {
        newVal = minimum() + ((maximum() - minimum() + 1) * (height() - e->y())) / height();
    }
    setValue(!invertedAppearance() ? newVal : maximum() - newVal);
}

class GradientSlider::Private
{
public:
    QVector<QPair<float, QColor>> colors;
};

GradientSlider::GradientSlider(QWidget* parent)
    : JumpableSlider(Qt::Horizontal, parent)
    , p(new Private)
{
}

void GradientSlider::setGradient(const QColor& startColor, const QColor& stopColor)
{
    setGradient({{0, startColor}, {1, stopColor}});
}

void GradientSlider::setGradient(const QVector<QPair<float, QColor>>& colors)
{
    if (colors.size() <= 1) {
        qWarning() << "ColorSlider::setGradient: colors size should >= 2";
        return;
    }

    p->colors = colors;

    QString ori;
    float x1, y1, x2, y2;
    if (orientation() == Qt::Horizontal) {
        ori = "horizontal";
        x1 = 0;
        y1 = 0;
        x2 = 1;
        y2 = 0;
    }
    else {
        ori = "vertical";
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 1;
    }

    QString gradientStyle;
    for (const auto& color : colors) {
        gradientStyle += QString(",stop:%1 %2").arg(color.first).arg(color.second.name());
    }

    auto style = QString("QSlider::groove:%1{background:qlineargradient(x1:%2,y1:%3,x2:%4,y2:%5 %6);}"
                         "QSlider::handle:%1{background:#5C5C5C;border:1px solid;width:6px}")
                     .arg(ori)
                     .arg(x1)
                     .arg(y1)
                     .arg(x2)
                     .arg(y2)
                     .arg(gradientStyle);

    setStyleSheet(style);
}

QVector<QPair<float, QColor>> GradientSlider::gradientColor() const
{
    return p->colors;
}

class ColorSpinHSlider::Private
{
public:
    QSpinBox* spinbox;
    GradientSlider* slider;

    Private(const QString& name, QWidget* parent)
    {
        QLabel* text = new QLabel(name, parent);
        text->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        spinbox = new QSpinBox(parent);
        spinbox->setButtonSymbols(QAbstractSpinBox::NoButtons);
        slider = new GradientSlider(parent);
        slider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        auto layout = new QHBoxLayout(parent);
        layout->setAlignment(Qt::AlignLeft);
        layout->setMargin(0);
        layout->addWidget(text, 1);
        layout->addWidget(spinbox, 1);
        layout->addWidget(slider, 8);

        connect(slider, &QSlider::valueChanged, spinbox, &QSpinBox::setValue);
        connect(spinbox, QOverload<int>::of(&QSpinBox::valueChanged), slider, &GradientSlider::setValue);
    }
};
ColorSpinHSlider::ColorSpinHSlider(const QString& name, QWidget* parent)
    : QWidget(parent)
    , p(new Private(name, this))
{
    connect(p->slider, &QSlider::valueChanged, this, &ColorSpinHSlider::valueChanged);
}

void ColorSpinHSlider::setGradient(const QColor& startColor, const QColor& stopColor)
{
    p->slider->setGradient(startColor, stopColor);
}

void ColorSpinHSlider::setGradient(const QVector<QPair<float, QColor>>& colors)
{
    p->slider->setGradient(colors);
}

void ColorSpinHSlider::setValue(double value)
{
    p->spinbox->setValue(value);
}

void ColorSpinHSlider::setRange(double min, double max)
{
    p->slider->setRange(min, max);
    p->spinbox->setRange(min, max);
}

QVector<QPair<float, QColor>> ColorSpinHSlider::gradientColor() const
{
    return p->slider->gradientColor();
}

//--------------------------------------------- color button -------------------------------------------------------
class ColorButton::Private
{
public:
    QPoint pressPos;
    QColor color;
    int bolderTopWidth = 0;
    int bolderBottomWidth = 0;
    int bolderLeftWidth = 0;
    int bolderRightWidth = 0;

    void updateStyle(QPushButton* btn)
    {
        int minWidth = ZenoStyle::dpiScaled(20);
        int minHeight = ZenoStyle::dpiScaled(20);
        auto style = QString("QPushButton{min-width:%1px;min-height:%2px;background-color:%3;"
                             "border-top:%4px solid;border-bottom:%5px solid;"
                             "border-left:%6px solid;border-right:%7px solid;}"
                             "QPushButton:pressed{border: 1px solid #ffd700;}")
                             .arg(minWidth)
                             .arg(minHeight)
                             .arg(color.name())
                             .arg(bolderTopWidth)
                             .arg(bolderBottomWidth)
                             .arg(bolderLeftWidth)
                             .arg(bolderRightWidth);
        btn->setStyleSheet(style);
    }
};

ColorButton::ColorButton(QWidget* parent)
    : QPushButton(parent)
    , p(new Private)
{
    setAcceptDrops(true);
    connect(this, &QPushButton::clicked, this, [this]() { emit colorClicked(p->color); });
}

void ColorButton::setColor(const QColor& color)
{
    p->color = color;
    p->updateStyle(this);
}

void ColorButton::setBolderWidth(int top, int bottom, int left, int right)
{
    p->bolderTopWidth = top;
    p->bolderBottomWidth = bottom;
    p->bolderLeftWidth = left;
    p->bolderRightWidth = right;
    p->updateStyle(this);
}

QColor ColorButton::color() const
{
    return p->color;
}

void ColorButton::mousePressEvent(QMouseEvent* e)
{
    p->pressPos = e->pos();
    QPushButton::mousePressEvent(e);
}

void ColorButton::mouseMoveEvent(QMouseEvent* e)
{
    if (e->buttons() & Qt::LeftButton) {
        if ((p->pressPos - e->pos()).manhattanLength() > QApplication::startDragDistance()) {
            QMimeData* mime = new QMimeData;
            mime->setColorData(p->color);
            QPixmap pix(width(), height());
            pix.fill(p->color);
            QDrag* drg = new QDrag(this);
            drg->setMimeData(mime);
            drg->setPixmap(pix);
            drg->exec(Qt::CopyAction);
            // need let pushbutton release
            QMouseEvent event(QEvent::MouseButtonRelease, e->pos(), Qt::LeftButton, Qt::LeftButton, 0);
            QApplication::sendEvent(this, &event);
        }
    }
}

void ColorButton::dragEnterEvent(QDragEnterEvent* e)
{
    if (qvariant_cast<QColor>(e->mimeData()->colorData()).isValid())
        e->accept();
    else
        e->ignore();
}

void ColorButton::dragLeaveEvent(QDragLeaveEvent*)
{
    if (hasFocus()) parentWidget()->setFocus();
}

void ColorButton::dropEvent(QDropEvent* e)
{
    auto color = qvariant_cast<QColor>(e->mimeData()->colorData());
    if (color.isValid()) {
        setColor(color);
        emit colorDroped(color);
        e->accept();
    }
    else {
        e->ignore();
    }
}

//--------------------------------------------- color palette ------------------------------------------------------
class ColorPalette::Private
{
public:
    int columnCount = 0;
    QGridLayout* layout = nullptr;
    QVector<QColor> colors;

    Private(int column, QScrollArea* parent)
    {
        columnCount = column;

        auto scrollWidget = new QWidget(parent);
        layout = new QGridLayout(scrollWidget);
        layout->setAlignment(Qt::AlignTop);
        layout->setSpacing(0);
        layout->setMargin(0);

        parent->setWidget(scrollWidget);
    }

    std::pair<int, int> getLayoutIndex(int index) { return {index / columnCount, index % columnCount}; }

    void updateLayout(int begin, int end)
    {
        for (int i = begin; i < end; ++i) {
            int row = i / columnCount;
            int col = i % columnCount;
            auto btn = qobject_cast<ColorButton*>(layout->itemAtPosition(row, col)->widget());
            btn->setColor(colors[i]);
        }
    }

    void updateBolder(int begin, int end)
    {
        int size = colors.size();
        for (int i = begin; i < end; ++i) {
            int row = i / columnCount;
            int col = i % columnCount;
            auto btn = qobject_cast<ColorButton*>(layout->itemAtPosition(row, col)->widget());
            int bolderLeftWidth = col == 0 ? 1 : 0;
            int bolderTopWidth = row == 0 ? 1 : 0;
            btn->setBolderWidth(bolderTopWidth, 1, bolderLeftWidth, 1);
        }
    }
};

ColorPalette::ColorPalette(int column, QWidget* parent)
    : QScrollArea(parent)
    , p(new Private(column, this))
{
    setWidgetResizable(true);
    setAcceptDrops(true);
}

void ColorPalette::addColor(const QColor& color)
{
    int index = p->colors.size();
    p->colors.push_back(color);

    auto btn = new ColorButton(this);
    btn->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    btn->setToolTip(tr("Ctrl + click to remove color"));
    connect(btn, &ColorButton::colorClicked, this, [this, index](const QColor& color) {
        if (QApplication::keyboardModifiers() == Qt::ControlModifier) {
            auto layoutIndex = p->getLayoutIndex(index);
            removeColor(layoutIndex.first, layoutIndex.second);
        }
        else {
            emit colorClicked(color);
        }
    });
    connect(btn, &ColorButton::colorDroped, this, [this, index](const QColor& color) {
        // update color at index
        p->colors[index] = color;
    });

    auto layoutIndex = p->getLayoutIndex(index);
    p->layout->addWidget(btn, layoutIndex.first, layoutIndex.second);

    p->updateLayout(index, p->colors.size());
    p->updateBolder(index, p->colors.size());
}

void ColorPalette::setColor(const QColor& color, int row, int column)
{
    int index = row * p->columnCount + column;
    p->colors[index] = color;
    p->updateLayout(index, index + 1);
}

void ColorPalette::removeColor(int row, int column)
{
    int size = p->colors.size();
    auto item = p->layout->takeAt(size - 1);
    if (item->widget()) {
        delete item->widget();
    }
    delete item;

    int index = row * p->columnCount + column;
    p->colors.remove(index);
    p->updateLayout(index, p->colors.size());
    p->updateBolder(index, p->colors.size());
}

QColor ColorPalette::colorAt(int row, int column) const
{
    if (column >= p->columnCount) {
        return QColor();
    }
    int index = row * p->columnCount + column;
    if (index >= p->colors.size()) {
        return QColor();
    }
    return p->colors[index];
}

QVector<QColor> ColorPalette::colors() const
{
    return p->colors;
}

void ColorPalette::dragEnterEvent(QDragEnterEvent* e)
{
    if (qvariant_cast<QColor>(e->mimeData()->colorData()).isValid())
        e->accept();
    else
        e->ignore();
}

void ColorPalette::dropEvent(QDropEvent* e)
{
    auto color = qvariant_cast<QColor>(e->mimeData()->colorData());
    if (color.isValid()) {
        addColor(color);
        e->accept();
    }
    else {
        e->ignore();
    }
}

//--------------------------------------------- color preview -------------------------------------------------------
class ColorPreview::Private
{
public:
    ColorButton* pbtnCurrent;
    ColorButton* pbtnPrevious;

    Private(const QColor& color, QWidget* parent)
        : pbtnCurrent(new ColorButton(parent))
        , pbtnPrevious(new ColorButton(parent))
    {
        // pbtnCurrent->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        // pbtnPrevious->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

        pbtnCurrent->setBolderWidth(1, 1, 0, 1);
        pbtnPrevious->setBolderWidth(1, 1, 1, 1);

        pbtnCurrent->setColor(color);
        pbtnPrevious->setColor(color);

        auto layout = new QHBoxLayout(parent);
        layout->setSpacing(0);
        layout->setMargin(0);
        layout->addWidget(pbtnPrevious);
        layout->addWidget(pbtnCurrent);
    }

    void setCurrent(const QColor& color) { pbtnCurrent->setColor(color); }
};

ColorPreview::ColorPreview(const QColor& color, QWidget* parent)
    : QWidget(parent)
    , p(new Private(color, this))
{
    // only emit when current color changed
    connect(p->pbtnCurrent, &ColorButton::colorDroped, this, &ColorPreview::currentColorChanged);
}

void ColorPreview::setCurrentColor(const QColor& color)
{
    p->setCurrent(color);
}

QColor ColorPreview::currentColor() const
{
    return p->pbtnCurrent->color();
}

QColor ColorPreview::previousColor() const
{
    return p->pbtnPrevious->color();
}

//------------------------------------------- color combo widget ---------------------------
class ColorComboWidget::Private
{
public:
    std::queue<colorcombo::ICombination*> combs;
    QHBoxLayout* hlayout = nullptr;
    QPushButton* switchBtn = nullptr;
    JumpableSlider* factorSlider = nullptr;
    QSpinBox* factorSpinbox = nullptr;

    Private(QWidget* parent)
    {
        factorSpinbox = new QSpinBox(parent);
        factorSlider = new JumpableSlider(Qt::Horizontal, parent);
        switchBtn = new QPushButton(parent);
        switchBtn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        factorSpinbox->setButtonSymbols(QAbstractSpinBox::NoButtons);

        auto layout = new QGridLayout(parent);
        layout->setMargin(0);
        hlayout = new QHBoxLayout();
        hlayout->setMargin(0);
        hlayout->setSpacing(0);
        layout->addLayout(hlayout, 0, 0, 1, 3);
        layout->addWidget(switchBtn, 0, 3, 1, 1);
        layout->addWidget(factorSpinbox, 1, 0, 1, 1);
        layout->addWidget(factorSlider, 1, 1, 1, 3);

        connect(factorSlider, &QSlider::valueChanged, factorSpinbox, &QSpinBox::setValue);
        connect(factorSpinbox, QOverload<int>::of(&QSpinBox::valueChanged), factorSlider, &QSlider::setValue);
    }
};

ColorComboWidget::ColorComboWidget(QWidget* parent)
    : QWidget(parent)
    , p(new Private(this))
{
    // dummy
    addCombination(new colorcombo::ICombination(this));
    switchCombination();

    connect(p->switchBtn, &QPushButton::clicked, this, &ColorComboWidget::switchCombination);
    connect(p->factorSpinbox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        auto comb = p->combs.front();
        comb->setValue(value);
        emit combinationChanged(comb);
    });
}

void ColorComboWidget::addCombination(colorcombo::ICombination* combo)
{
    p->combs.push(combo);
}

void ColorComboWidget::clearCombination()
{
    while (!p->combs.empty()) {
        p->combs.pop();
    }
    // dummy
    addCombination(new colorcombo::ICombination(this));
    switchCombination();
}

colorcombo::ICombination* ColorComboWidget::currentCombination() const
{
    return p->combs.front();
}

void ColorComboWidget::setColors(const QVector<QColor>& colors)
{
    for (int i = 0; i < colors.size(); ++i) {
        auto btn = qobject_cast<ColorButton*>(p->hlayout->itemAt(i)->widget());
        btn->setColor(colors[i]);
    }
}

void ColorComboWidget::switchCombination()
{
    if (p->combs.empty()) return;

    auto front = p->combs.front();
    p->combs.pop();
    p->combs.push(front);

    auto currentComb = p->combs.front();

    // clear
    QLayoutItem* item;
    while (item = p->hlayout->takeAt(0)) {
        if (item->widget()) {
            delete item->widget();
        }
        delete item;
    }
    // add
    auto colors = currentComb->genColors(Qt::white);
    int size = colors.size() + 1;
    for (int i = 0; i < size; ++i) {
        auto btn = new ColorButton(this);
        int bolderRightWidth = i < size - 1 ? 0 : 1;
        btn->setBolderWidth(1, 1, 1, bolderRightWidth);
        btn->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        btn->setAcceptDrops(false); // color can't be changed by drop
        connect(btn, &ColorButton::colorClicked, this, &ColorComboWidget::colorClicked);
        p->hlayout->addWidget(btn);
    }

    p->factorSlider->blockSignals(true);
    p->factorSpinbox->blockSignals(true);
    p->factorSlider->setRange(currentComb->min(), currentComb->max());
    p->factorSpinbox->setRange(currentComb->min(), currentComb->max());
    p->factorSlider->setValue(currentComb->getValue());
    p->factorSpinbox->setValue(currentComb->getValue());
    p->factorSlider->setEnabled(currentComb->rangeEnabled());
    p->factorSpinbox->setEnabled(currentComb->rangeEnabled());
    p->factorSlider->blockSignals(false);
    p->factorSpinbox->blockSignals(false);

    emit combinationChanged(currentComb);
}

//------------------------------------------ color lineedit --------------------------------

ColorLineEdit::ColorLineEdit(QWidget* parent)
    : QLineEdit(parent)
{
    connect(this, &ColorLineEdit::editingFinished, this, [this]() {
        setText(text().toUpper());
        emit currentColorChanged(QColor(text()));
    });
}
void ColorLineEdit::setColor(const QColor& color)
{
    setText(color.name().toUpper());
}

//------------------------------------------ color picker ----------------------------------
class ColorPicker::Private
{
public:
    int rectLength = 20;
    int scaleSize = 10;
    QPoint cursorPos;
    QImage fullScreenImg;

    void grabFullScreen()
    {
        const QDesktopWidget* desktop = QApplication::desktop();
        const QPixmap pixmap = QApplication::primaryScreen()->grabWindow(desktop->winId(), desktop->pos().x(), desktop->pos().y(),
                                                                         desktop->width(), desktop->height());
        fullScreenImg = pixmap.toImage();
    }

    QRect getScreenRect() const
    {
        const QDesktopWidget* desktop = QApplication::desktop();
        return QRect(desktop->pos(), desktop->size());
    }

    QColor getColorAt(QPoint p) const { return fullScreenImg.pixelColor(p); }

    QImage getScaledImage(QPoint p) const
    {
        int rectHalfLength = rectLength / 2;
        QImage img = fullScreenImg.copy(p.x() - rectHalfLength, p.y() - rectHalfLength, rectLength, rectLength);
        return img.scaled(scaleSize * rectLength, scaleSize * rectLength);
    }

    QScreen* getScreenAt(QPoint p) const
    {
#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
        QScreen* screen = QApplication::screenAt(p);
#else
        int screenNum = QApplication::desktop()->screenNumber(p);
        QScreen* screen = QApplication::screens().at(screenNum);
#endif
        return screen;
    }
};

ColorPicker::ColorPicker(QWidget* parent)
    : QWidget(parent)
    , p(new Private)
{
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
}

QColor ColorPicker::grabScreenColor(QPoint p) const
{
    // not use now, just make screenshot and get color from it
    const QDesktopWidget* desktop = QApplication::desktop();
    const QPixmap pixmap = QApplication::primaryScreen()->grabWindow(desktop->winId(), p.x(), p.y(), 1, 1);
    QImage i = pixmap.toImage();
    return i.pixel(0, 0);
}

void ColorPicker::startColorPicking()
{
    p->cursorPos = QCursor::pos();
    p->grabFullScreen();
    setGeometry(p->getScreenRect());
    showFullScreen();
    setFocus();
}

void ColorPicker::releaseColorPicking()
{
    hide();
}

void ColorPicker::paintEvent(QPaintEvent* e)
{
    QPainter painter(this);
    // background
    painter.drawImage(0, 0, p->fullScreenImg);
    // scaled img
    auto img = p->getScaledImage(p->cursorPos);
    auto currentColor = p->getColorAt(p->cursorPos);
    auto screen = p->getScreenAt(p->cursorPos);
    auto rect = screen->geometry();
    // calculate img pos
    int dx = 20, dy = 20;
    int x, y;
    if (rect.right() - p->cursorPos.x() < img.width() + dx) {
        x = p->cursorPos.x() - img.width() - dx;
    }
    else {
        x = p->cursorPos.x() + dx;
    }
    if (rect.height() - p->cursorPos.y() < img.height() + dy) {
        y = p->cursorPos.y() - img.height() + dy;
    }
    else {
        y = p->cursorPos.y() + dy;
    }

    painter.translate(x, y);
    painter.drawImage(0, 0, img);

    int rectWidth = 10;
    int halfRectWidth = rectWidth / 2;
    int halfH = img.height() / 2;
    int halfW = img.width() / 2;
    // cross
    painter.setPen(QPen(QColor("#aadafa7f"), rectWidth));
    painter.drawLine(halfW, halfRectWidth, halfW, halfH - rectWidth);
    painter.drawLine(halfW, rectWidth + halfH, halfW, img.height() - halfRectWidth);
    painter.drawLine(halfRectWidth, halfH, halfW - rectWidth, halfH);
    painter.drawLine(rectWidth + halfW, halfH, img.width() - halfRectWidth, halfH);
    // bolder
    painter.setPen(QPen(qGray(currentColor.rgb()) > 127 ? Qt::black : Qt::white, 1));
    painter.drawRect(0, 0, img.width(), img.height());
    painter.drawRect(halfW - halfRectWidth, halfH - halfRectWidth, rectWidth, rectWidth);
}

void ColorPicker::mouseMoveEvent(QMouseEvent* e)
{
    p->cursorPos = e->pos();
    update();
}

void ColorPicker::mouseReleaseEvent(QMouseEvent* e)
{
    if (e->button() == Qt::LeftButton) {
        emit colorSelected(p->getColorAt(QCursor::pos()));
        releaseColorPicking();
    }
    else if (e->button() == Qt::RightButton) {
        releaseColorPicking();
    }
}

void ColorPicker::keyPressEvent(QKeyEvent* e)
{
    switch (e->key()) {
        case Qt::Key_Escape:
            releaseColorPicking();
            break;
        case Qt::Key_Return:
        case Qt::Key_Enter:
            emit colorSelected(p->getColorAt(QCursor::pos()));
            releaseColorPicking();
            break;
        case Qt::Key_Up:
            QCursor::setPos(p->cursorPos.x(), p->cursorPos.y() - 1);
            break;
        case Qt::Key_Down:
            QCursor::setPos(p->cursorPos.x(), p->cursorPos.y() + 1);
            break;
        case Qt::Key_Left:
            QCursor::setPos(p->cursorPos.x() - 1, p->cursorPos.y());
            break;
        case Qt::Key_Right:
            QCursor::setPos(p->cursorPos.x() + 1, p->cursorPos.y());
            break;
        default:
            break;
    }
}

void ColorPicker::focusOutEvent(QFocusEvent* e)
{
    releaseColorPicking();
}

//------------------------------------------------------- color data --------------------------------------------
struct ColorEditorData
{
    static constexpr int rowCount = 4;
    static constexpr int colCount = 12;
    QColor standardColor[rowCount * colCount];

    ColorEditorData()
    {
        // standard
        int i = 0;
        for (int s = 0; s < rowCount; ++s) {
            for (int h = 0; h < colCount; ++h) {
                standardColor[i++] = QColor::fromHsvF(1.0 * h / colCount, 1.0 - 1.0 * s / rowCount, 1.0);
            }
        }
    }

    QVector<QColor> readSettings()
    {
        const QSettings settings(QSettings::UserScope, QStringLiteral("__ColorEditor_4x12"));
        int count = settings.value(QLatin1String("customCount")).toInt();
        QVector<QColor> customColor(count);
        // if zero, init with standard
        if (count == 0) {
            for (const auto& color : standardColor) {
                customColor.append(color);
            }
        }
        // otherwise, init with settings
        else {
            for (int i = 0; i < count; ++i) {
                const QVariant v = settings.value(QLatin1String("customColors/") + QString::number(i));
                if (v.isValid()) {
                    customColor[i] = v.toUInt();
                }
            }
        }

        return customColor;
    }
    void writeSettings(const QVector<QColor>& colors)
    {
        QSettings settings(QSettings::UserScope, QStringLiteral("__ColorEditor_4x12"));
        int count = colors.size();
        settings.setValue(QLatin1String("customCount"), count);
        for (int i = 0; i < count; ++i) {
            settings.setValue(QLatin1String("customColors/") + QString::number(i), colors[i].rgb());
        }
    }
};

//------------------------------------------ color editor ----------------------------------
class ColorEditor::Private
{
public:
    ColorWheel* wheel;
    ColorLineEdit* colorText;
    ColorPreview* preview;
    ColorPicker* picker;
    QPushButton* pickerBtn;
    ColorComboWidget* combo;
    QGroupBox* previewGroup;
    QGroupBox* comboGroup;
    ColorPalette* palette;
    ColorSpinHSlider* rSlider;
    ColorSpinHSlider* gSlider;
    ColorSpinHSlider* bSlider;
    ColorSpinHSlider* hSlider;
    ColorSpinHSlider* sSlider;
    ColorSpinHSlider* vSlider;

    QColor currentColor;
    QColor selectedColor;
    ColorEditorData colorData;

    Private(const QColor& color, QDialog* parent)
    {
        selectedColor = color;
        // left
        picker = new ColorPicker(parent);
        pickerBtn = new QPushButton(parent);
        wheel = new ColorWheel(parent);
        colorText = new ColorLineEdit(parent);
        preview = new ColorPreview(color, parent);
        combo = new ColorComboWidget(parent);
        previewGroup = new QGroupBox(tr("Previous/Current Colors"), parent);
        comboGroup = new QGroupBox(tr("Color Combination"), parent);

        colorText->setMaximumWidth(ZenoStyle::dpiScaled(60));
        colorText->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        pickerBtn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

        auto previewWidget = new QWidget(parent);
        auto previewLayout = new QHBoxLayout(previewWidget);
        previewLayout->setMargin(0);
        previewLayout->addWidget(preview);
        previewLayout->addWidget(pickerBtn);

        auto previewGroupLayout = new QHBoxLayout(previewGroup);
        previewGroupLayout->addWidget(previewWidget);

        auto comboGroupLayout = new QHBoxLayout(comboGroup);
        comboGroupLayout->addWidget(combo);

        auto leftWidget = new QWidget(parent);
        auto leftLayout = new QGridLayout(leftWidget);
        leftLayout->setContentsMargins(0, 0, 5, 0);
        leftLayout->setSpacing(0);
        leftLayout->addWidget(wheel, 0, 0);
        leftLayout->addWidget(colorText, 1, 0, Qt::AlignRight);
        leftLayout->addWidget(previewGroup, 2, 0);
        leftLayout->addWidget(comboGroup, 3, 0);

        // right
        palette = new ColorPalette(colorData.colCount, parent);
        rSlider = new ColorSpinHSlider("R", parent);
        gSlider = new ColorSpinHSlider("G", parent);
        bSlider = new ColorSpinHSlider("B", parent);
        hSlider = new ColorSpinHSlider("H", parent);
        sSlider = new ColorSpinHSlider("S", parent);
        vSlider = new ColorSpinHSlider("V", parent);

        auto colorSlider = new QWidget(parent);
        auto colorSliderLayout = new QVBoxLayout(colorSlider);
        colorSliderLayout->setContentsMargins(5, 0, 0, 0);
        colorSliderLayout->setSpacing(2);
        colorSliderLayout->addWidget(rSlider);
        colorSliderLayout->addWidget(gSlider);
        colorSliderLayout->addWidget(bSlider);
        colorSliderLayout->addSpacing(5);
        colorSliderLayout->addWidget(hSlider);
        colorSliderLayout->addWidget(sSlider);
        colorSliderLayout->addWidget(vSlider);

        rSlider->setRange(0, 255);
        gSlider->setRange(0, 255);
        bSlider->setRange(0, 255);
        hSlider->setRange(0, 359);
        sSlider->setRange(0, 255);
        vSlider->setRange(0, 255);

        setGradientR(color);
        setGradientG(color);
        setGradientB(color);
        setGradientH(color);
        setGradientS(color);
        setGradientV(color);

        auto rightSplitter = new QSplitter(Qt::Vertical, parent);
        rightSplitter->addWidget(palette);
        rightSplitter->addWidget(colorSlider);
        rightSplitter->setCollapsible(0, false);
        rightSplitter->setCollapsible(1, false);
        auto equalH = std::max(palette->minimumSizeHint().height(), colorSlider->minimumSizeHint().height());
        rightSplitter->setSizes({equalH * 2, equalH * 1}); // setStretchFactor not always work well

        auto mainSplitter = new QSplitter(parent);
        mainSplitter->addWidget(leftWidget);
        mainSplitter->addWidget(rightSplitter);
        mainSplitter->setStretchFactor(0, 3);
        mainSplitter->setStretchFactor(1, 7);
        mainSplitter->setCollapsible(0, false);
        mainSplitter->setCollapsible(1, false);
        // buttons
        auto buttons = new QDialogButtonBox(parent);
        auto okBtn = buttons->addButton(QDialogButtonBox::Ok);
        auto cancleBtn = buttons->addButton(QDialogButtonBox::Cancel);
        connect(okBtn, &QPushButton::clicked, parent, [this, parent]() {
            selectedColor = currentColor;
            parent->accept();
        });
        connect(cancleBtn, &QPushButton::clicked, parent, &QDialog::reject);

        auto layout = new QVBoxLayout(parent);
        layout->setMargin(5);
        layout->addWidget(mainSplitter);
        layout->addWidget(buttons);
    }

    void blockColorSignals(bool block)
    {
        wheel->blockSignals(block);
        colorText->blockSignals(block);
        preview->blockSignals(block);
        combo->blockSignals(block);
        palette->blockSignals(block);
        rSlider->blockSignals(block);
        gSlider->blockSignals(block);
        bSlider->blockSignals(block);
        hSlider->blockSignals(block);
        sSlider->blockSignals(block);
        vSlider->blockSignals(block);
    }

    void setGradient(const QColor& color)
    {
        bool rChanged = color.red() != currentColor.red();
        bool gChanged = color.green() != currentColor.green();
        bool bChanged = color.blue() != currentColor.blue();
        bool hChanged = color.hsvHue() != currentColor.hsvHue();
        bool sChanged = color.hsvSaturation() != currentColor.hsvSaturation();
        bool vChanged = color.value() != currentColor.value();

        if (gChanged || bChanged) {
            setGradientR(color);
        }
        if (rChanged || bChanged) {
            setGradientG(color);
        }
        if (rChanged || gChanged) {
            setGradientB(color);
        }
        if (sChanged || vChanged) {
            setGradientH(color);
        }
        if (hChanged || vChanged) {
            setGradientS(color);
        }
        if (hChanged || sChanged) {
            setGradientV(color);
        }
    }

    void setGradientR(const QColor& color)
    {
        rSlider->setGradient(QColor(0, color.green(), color.blue()), QColor(255, color.green(), color.blue()));
    }
    void setGradientG(const QColor& color)
    {
        gSlider->setGradient(QColor(color.red(), 0, color.blue()), QColor(color.red(), 255, color.blue()));
    }
    void setGradientB(const QColor& color)
    {
        bSlider->setGradient(QColor(color.red(), color.green(), 0), QColor(color.red(), color.green(), 255));
    }
    void setGradientH(const QColor& color)
    {
        // hSlider is unique
        static QVector<QPair<float, QColor>> hColors(7);
        for (int i = 0; i < hColors.size(); ++i) {
            float f = 1.0 * i / (hColors.size() - 1);
            hColors[i] = {f, QColor::fromHsvF(f, color.hsvSaturationF(), color.valueF())};
        }
        hSlider->setGradient(hColors);
    }
    void setGradientS(const QColor& color)
    {
        sSlider->setGradient(QColor::fromHsvF(color.hsvHueF(), 0, color.valueF()), QColor::fromHsvF(color.hsvHueF(), 1, color.valueF()));
    }
    void setGradientV(const QColor& color)
    {
        vSlider->setGradient(QColor::fromHsvF(color.hsvHueF(), color.hsvSaturationF(), 0),
                             QColor::fromHsvF(color.hsvHueF(), color.hsvSaturationF(), 1));
    }
};

ColorEditor::ColorEditor(QWidget* parent)
    : ColorEditor(Qt::white, parent)
{
}

ColorEditor::ColorEditor(const QColor& initial, QWidget* parent)
    : QDialog(parent)
    , p(new Private(initial, this))
{
    setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    setWindowTitle(tr("ColorEditor"));
    setMinimumSize(ZenoStyle::dpiScaled(500), ZenoStyle::dpiScaled(350));
    initSlots();
    // init combinations
    p->combo->addCombination(new colorcombo::Analogous(this));
    p->combo->addCombination(new colorcombo::Complementary(this));
    p->combo->addCombination(new colorcombo::Monochromatic(this));
    p->combo->addCombination(new colorcombo::Triadic(this));
    p->combo->addCombination(new colorcombo::Tetradic(this));
    // init colors for palette
    auto paletteColors = p->colorData.readSettings();
    for (const auto& color : paletteColors) {
        p->palette->addColor(color);
    }
    // current combination
    p->wheel->setColorCombination(p->combo->currentCombination());
    // current color
    setCurrentColor(initial);
}

void ColorEditor::setCurrentColor(const QColor& color)
{
    p->blockColorSignals(true);
    {
        p->wheel->setSelectedColor(color);
        p->colorText->setColor(color);
        p->preview->setCurrentColor(color);
        p->setGradient(color);
        p->rSlider->setValue(color.red());
        p->gSlider->setValue(color.green());
        p->bSlider->setValue(color.blue());
        p->hSlider->setValue(color.hsvHue());
        p->sSlider->setValue(color.hsvSaturation());
        p->vSlider->setValue(color.value());
    }
    p->blockColorSignals(false);

    p->currentColor = color;
}

QColor ColorEditor::currentColor() const
{
    return p->currentColor;
}

QColor ColorEditor::selectedColor() const
{
    return p->selectedColor;
}

void ColorEditor::setColorCombinations(const QVector<colorcombo::ICombination*> combinations)
{
    p->combo->clearCombination();
    for (const auto& combination : combinations) {
        p->combo->addCombination(combination);
    }
}

void ColorEditor::closeEvent(QCloseEvent* e)
{
    // save colors on close
    p->colorData.writeSettings(p->palette->colors());
    QDialog::closeEvent(e);
}

void ColorEditor::keyPressEvent(QKeyEvent* e)
{
    if (e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return) {
        return;
    }
    QDialog::keyPressEvent(e);
}

void ColorEditor::initSlots()
{
    // picker
    connect(p->pickerBtn, &QPushButton::clicked, p->picker, &ColorPicker::startColorPicking);
    connect(p->picker, &ColorPicker::colorSelected, this, &ColorEditor::setCurrentColor);
    // color combination
    connect(p->wheel, &ColorWheel::combinationColorChanged, p->combo, &ColorComboWidget::setColors);
    connect(p->combo, &ColorComboWidget::combinationChanged, this, [this](colorcombo::ICombination* combination) {
        p->wheel->setColorCombination(combination);
        p->comboGroup->setTitle(combination->name());
    });
    // color wheel/text/preview/combo
    connect(p->wheel, &ColorWheel::colorSelected, this, &ColorEditor::setCurrentColor);
    connect(p->colorText, &ColorLineEdit::currentColorChanged, this, &ColorEditor::setCurrentColor);
    connect(p->preview, &ColorPreview::currentColorChanged, this, &ColorEditor::setCurrentColor);
    connect(p->palette, &ColorPalette::colorClicked, this, &ColorEditor::setCurrentColor);
    connect(p->combo, &ColorComboWidget::colorClicked, this, [this](const QColor& color) {
        // don't change wheel color
        p->wheel->setEnabled(false);
        setCurrentColor(color);
        p->wheel->setEnabled(true);
    });
    // color slider
    connect(p->rSlider, &ColorSpinHSlider::valueChanged, this, [this](int value) {
        auto color = QColor(value, p->currentColor.green(), p->currentColor.blue());
        setCurrentColor(color);
    });
    connect(p->gSlider, &ColorSpinHSlider::valueChanged, this, [this](int value) {
        auto color = QColor(p->currentColor.red(), value, p->currentColor.blue());
        setCurrentColor(color);
    });
    connect(p->bSlider, &ColorSpinHSlider::valueChanged, this, [this](int value) {
        auto color = QColor(p->currentColor.red(), p->currentColor.green(), value);
        setCurrentColor(color);
    });
    connect(p->hSlider, &ColorSpinHSlider::valueChanged, this, [this](int value) {
        auto color = QColor::fromHsv(value, p->currentColor.hsvSaturation(), p->currentColor.value());
        setCurrentColor(color);
    });
    connect(p->sSlider, &ColorSpinHSlider::valueChanged, this, [this](int value) {
        auto color = QColor::fromHsv(p->currentColor.hsvHue(), value, p->currentColor.value());
        setCurrentColor(color);
    });
    connect(p->vSlider, &ColorSpinHSlider::valueChanged, this, [this](int value) {
        auto color = QColor::fromHsv(p->currentColor.hsvHue(), p->currentColor.hsvSaturation(), value);
        setCurrentColor(color);
    });
}

QColor ColorEditor::getColor(const QColor& initial, QWidget* parent, const QString& title)
{
    ColorEditor dlg(initial, parent);
    if (!title.isEmpty()) dlg.setWindowTitle(title);
    dlg.exec();
    return dlg.selectedColor();
}
