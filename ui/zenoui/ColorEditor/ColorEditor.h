#pragma once

#include <memory>

#include <QDialog>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QWidget>

//------------------------------------------- color correction -----------------------------------------------
struct ColorCorrection
{
    float gamma = 2.2f;
    void correct(QColor& color);
    void correct(QImage& image);
};

//------------------------------------------- color combination ----------------------------------------------
namespace colorcombo
{
class ICombination : public QObject
{
    Q_OBJECT
public:
    explicit ICombination(QObject* parent = nullptr);
    explicit ICombination(double min, double max, double value, int decimals, bool rangeEnabled, QObject* parent = nullptr);
    virtual ~ICombination() = default;
    virtual QString name();
    virtual QVector<QColor> genColors(const QColor& color);
    void setRange(double min, double max);
    void setValue(double value);
    void setDecimals(int decimals);
    double min() const;
    double max() const;
    double getValue() const;
    bool rangeEnabled() const;
    int decimals() const;

private:
    double m_min;
    double m_max;
    double m_value;
    int m_decimals;
    bool m_rangeEnabled;
};

class Complementary : public ICombination
{
public:
    explicit Complementary(QObject* parent = nullptr);
    virtual QString name() override;
    virtual QVector<QColor> genColors(const QColor& color) override;
};

class Monochromatic : public ICombination
{
public:
    explicit Monochromatic(QObject* parent = nullptr);
    virtual QString name() override;
    virtual QVector<QColor> genColors(const QColor& color) override;
};

class Analogous : public ICombination
{
public:
    explicit Analogous(QObject* parent = nullptr);
    virtual QString name() override;
    virtual QVector<QColor> genColors(const QColor& color) override;
};

class Triadic : public ICombination
{
public:
    explicit Triadic(QObject* parent = nullptr);
    virtual QString name() override;
    virtual QVector<QColor> genColors(const QColor& color) override;
};

class Tetradic : public ICombination
{
public:
    explicit Tetradic(QObject* parent = nullptr);
    virtual QString name() override;
    virtual QVector<QColor> genColors(const QColor& color) override;
};
} // namespace colorcombo

//-------------------------------------------------- color wheel --------------------------------------------------
class ColorWheel : public QWidget
{
    Q_OBJECT
public:
    explicit ColorWheel(QWidget* parent = nullptr);
    ~ColorWheel();

    void setColorCombination(colorcombo::ICombination* combination);
    void setSelectedColor(const QColor& color);
    void setColorCorrection(ColorCorrection* colorCorrection);
    QColor getSelectedColor() const;
    QColor getColor(int x, int y) const;

signals:
    void colorSelected(const QColor& color);
    void combinationColorChanged(const QVector<QColor>& colors);

protected:
    void paintEvent(QPaintEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

private:
    void processMouseEvent(QMouseEvent* e);
    void drawSelector(QPainter* painter, const QColor& color, int radius);

    class Private;
    std::unique_ptr<Private> p;
};

//---------------------------------------------- color slider -------------------------------------------------------
class MixedSpinBox : public QDoubleSpinBox
{
public:
    explicit MixedSpinBox(QWidget* parent = nullptr);
    virtual QString textFromValue(double value) const override;

protected:
    void keyPressEvent(QKeyEvent* e) override;
};

class JumpableSlider : public QSlider
{
    Q_OBJECT
public:
    explicit JumpableSlider(QWidget* parent);
    explicit JumpableSlider(Qt::Orientation orientation, QWidget* parent = nullptr);
    ~JumpableSlider();

    void setValue(double value);
    void setMinimum(double value);
    void setMaximum(double value);
    void setRange(double minValue, double maxValue);
    void setSingleStep(double value);

    double value() const;
    double minimum() const;
    double maximum() const;
    double singleStep() const;

signals:
    void valueChanged(double value);

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;

private:
    void handleMouseEvent(QMouseEvent* e);

    class Private;
    std::unique_ptr<Private> p;
};

class GradientSlider : public JumpableSlider
{
    Q_OBJECT
public:
    explicit GradientSlider(QWidget* parent = nullptr);
    ~GradientSlider();

    void setGradient(const QColor& startColor, const QColor& stopColor);
    void setGradient(const QGradientStops& colors);
    void setColorCorrection(ColorCorrection* colorCorrection);
    QGradientStops gradientColor() const;

protected:
    void paintEvent(QPaintEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

private:
    class Private;
    std::unique_ptr<Private> p;
};

class ColorSpinHSlider : public QWidget
{
    Q_OBJECT
public:
    explicit ColorSpinHSlider(const QString& name, QWidget* parent = nullptr);
    ~ColorSpinHSlider();

    void setGradient(const QColor& startColor, const QColor& stopColor);
    void setGradient(const QGradientStops& colors);
    void setColorCorrection(ColorCorrection* colorCorrection);
    void setValue(double value);
    void setRange(double min, double max);
    QGradientStops gradientColor() const;
    double value() const;

signals:
    void valueChanged(double value);

private:
    class Private;
    std::unique_ptr<Private> p;
};

//--------------------------------------------- color button -------------------------------------------------------
class ColorButton : public QPushButton
{
    Q_OBJECT
public:
    explicit ColorButton(QWidget* parent = nullptr);
    ~ColorButton();

    void setColor(const QColor& color);
    void setColorCorrection(ColorCorrection* colorCorrection);
    void setBolderWidth(int top, int bottom, int left, int right);
    QColor color() const;

signals:
    void colorClicked(const QColor& color);
    void colorDroped(const QColor& color);

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dragLeaveEvent(QDragLeaveEvent*) override;
    void dropEvent(QDropEvent* e) override;

private:
    class Private;
    std::unique_ptr<Private> p;
};

//--------------------------------------------- color palette ------------------------------------------------------
class ColorPalette : public QScrollArea
{
    Q_OBJECT
public:
    explicit ColorPalette(int column, QWidget* parent = nullptr);
    ~ColorPalette();

    void addColor(const QColor& color);
    void setColor(const QColor& color, int row, int column);
    void removeColor(int row, int column);
    void setColorCorrection(ColorCorrection* colorCorrection);
    QColor colorAt(int row, int column) const;
    QVector<QColor> colors() const;

signals:
    void colorClicked(const QColor& color);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;

private:
    class Private;
    std::unique_ptr<Private> p;
};

//--------------------------------------------- color preview -------------------------------------------------------
class ColorPreview : public QWidget
{
    Q_OBJECT
public:
    explicit ColorPreview(const QColor& color, QWidget* parent = nullptr);
    ~ColorPreview();

    void setCurrentColor(const QColor& color);
    void setColorCorrection(ColorCorrection* colorCorrection);
    QColor currentColor() const;
    QColor previousColor() const;

signals:
    void currentColorChanged(const QColor& color);

private:
    class Private;
    std::unique_ptr<Private> p;
};

//------------------------------------------- color combo widget ---------------------------
class ColorComboWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ColorComboWidget(QWidget* parent = nullptr);
    ~ColorComboWidget();

    void addCombination(colorcombo::ICombination* combo);
    void clearCombination();
    void switchCombination();
    void setColors(const QVector<QColor>& colors);
    void setColorCorrection(ColorCorrection* colorCorrection);
    colorcombo::ICombination* currentCombination() const;

signals:
    void colorClicked(const QColor& color);
    void combinationChanged(colorcombo::ICombination* combo);

private:
    class Private;
    std::unique_ptr<Private> p;
};

//------------------------------------------ color lineedit --------------------------------
class ColorLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    explicit ColorLineEdit(QWidget* parent = nullptr);
    void setColor(const QColor& color);

signals:
    void currentColorChanged(const QColor& color);

protected:
    void keyPressEvent(QKeyEvent* e) override;
};

//------------------------------------------ color picker ----------------------------------
class ColorPicker : public QWidget
{
    Q_OBJECT
public:
    explicit ColorPicker(QWidget* parent = nullptr);
    ~ColorPicker();

    QColor grabScreenColor(QPoint p) const;
    void startColorPicking();
    void releaseColorPicking();

signals:
    void colorSelected(const QColor& color);

protected:
    void paintEvent(QPaintEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    void focusOutEvent(QFocusEvent* e) override;

private:
    class Private;
    std::unique_ptr<Private> p;
};

//------------------------------------------ color editor ----------------------------------
class ColorEditor : public QDialog
{
    Q_OBJECT
public:
    explicit ColorEditor(QWidget* parent = nullptr);
    explicit ColorEditor(const QColor& initial, QWidget* parent = nullptr);
    ~ColorEditor();

    static QColor getColor(const QColor& initial, QWidget* parent = nullptr, const QString& title = "");

    void setCurrentColor(const QColor& color);
    QColor currentColor() const;
    QColor selectedColor() const;

    void setColorCombinations(const QVector<colorcombo::ICombination*> combinations);

signals:
    void currentColorChanged(const QColor& color);

protected:
    void closeEvent(QCloseEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private:
    void initSlots();

    class Private;
    std::unique_ptr<Private> p;
};