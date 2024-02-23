//
// Created by zh on 2024/2/22.
//

#include "ZOptixCameraSetting.h"

ZOptixCameraSetting::ZOptixCameraSetting(zenovis::ZOptixCameraSettingInfo &info, QWidget *parent) : QDialog(parent) {
    QPushButton *okButton = new QPushButton("OK", this);
    QPushButton *cancelButton = new QPushButton("Cancel", this);

    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);

    QVBoxLayout *mainLayout = new QVBoxLayout;

    QDoubleSpinBox* m_aperture = new QDoubleSpinBox();
    m_aperture->setDecimals(4);
    m_aperture->setRange(0.0, 10000);
    m_aperture->setValue(info.aperture);

    QDoubleSpinBox* m_shutter_speed = new QDoubleSpinBox();
    m_shutter_speed->setDecimals(4);
    m_shutter_speed->setRange(0.0, 10000);
    m_shutter_speed->setValue(info.shutter_speed);

    QComboBox *m_iso = new QComboBox(this);
    m_iso->setEditable(true);
    m_iso->addItem("100");
    m_iso->addItem("150");
    m_iso->addItem("200");
    m_iso->addItem("400");
    m_iso->addItem("800");
    m_iso->addItem("1600");
    m_iso->addItem("3200");
    m_iso->addItem("6400");
    m_iso->setEditText(QString::number(info.iso));

    QCheckBox *m_aces = new QCheckBox(tr("ACES"));
    m_aces->setStyleSheet("color: white;");
    m_aces->setChecked(info.aces);

    QCheckBox *m_exposure = new QCheckBox(tr("Exposure"));
    m_exposure->setStyleSheet("color: white;");
    m_exposure->setChecked(info.exposure);

    mainLayout->addWidget(new QLabel("Aperture"));
    mainLayout->addWidget(m_aperture);
    mainLayout->addWidget(new QLabel("ShutterSpeed"));
    mainLayout->addWidget(m_shutter_speed);
    mainLayout->addWidget(new QLabel("ISO"));
    mainLayout->addWidget(m_iso);
    mainLayout->addWidget(m_aces);
    mainLayout->addWidget(m_exposure);

    mainLayout->addLayout(buttonLayout);

    // 设置布局
    setLayout(mainLayout);

    // 连接按钮信号到槽
    connect(okButton, &QPushButton::clicked, this, &ZOptixCameraSetting::accept);
    connect(cancelButton, &QPushButton::clicked, this, &ZOptixCameraSetting::reject);

    connect(m_aperture, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [&](double value) {
        info.aperture = value;
    });
    connect(m_shutter_speed, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [&](double value) {
        info.shutter_speed = value;
    });

    connect(m_iso, &QComboBox::currentTextChanged, this, [&](const QString &text) {
        info.iso = text.toFloat();
    });
    connect(m_aces, &QCheckBox::stateChanged, this, [&](int state) {
        info.aces = state == Qt::Checked;
    });
    connect(m_exposure, &QCheckBox::stateChanged, this, [&](int state) {
        info.exposure = state == Qt::Checked;
    });

}
