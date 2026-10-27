#include "zpreferencesdlg.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zpathedit.h>
#include "settings/zenosettingsmanager.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"
#include "zshortcutsettingdlg.h"
#include <zenoui/comctrl/zlabel.h>
#include "viewport/displaywidget.h"
#include <zenovis/DrawOptions.h>

//Language Pane
ZLanguagePane::ZLanguagePane(QWidget* parent)
    :QWidget(parent)
{
    bool bChinese = ZenoSettingsManager::GetInstance().getValue(zsUseChinese).toBool();
    QHBoxLayout* pLayout = new QHBoxLayout(this);
    m_pEngBtn = new QRadioButton(tr("English"), this);
    m_pCNBtn = new QRadioButton(tr("Chinese"), this);
    m_pCNBtn->setChecked(bChinese);
    m_pEngBtn->setChecked(!bChinese);
    pLayout->addWidget(m_pEngBtn);
    pLayout->addWidget(m_pCNBtn);
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
}

void ZLanguagePane::saveValue()
{
    bool value = m_pCNBtn->isChecked();
    auto& inst = ZenoSettingsManager::GetInstance();
    bool bChinese = inst.getValue(zsUseChinese).toBool();
    if (inst.setValue(zsUseChinese, value))
    {
        QMessageBox msg(QMessageBox::Information, tr("Language"),
            tr("Please restart Zeno to apply changes."),
            QMessageBox::Ok, this);
        msg.exec();
    }
}

//NASLOC Pane
ZNASLOCPane::ZNASLOCPane(QWidget* parent)
    :QWidget(parent)
{
    const QString& path = ZenoSettingsManager::GetInstance().getValue(zsNASLOC).toString();
    QHBoxLayout* pLayout = new QHBoxLayout(this);
    CALLBACK_SWITCH cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn); //deal with ubuntu dialog slow problem when update viewport.
    };
    m_pPthEdit = new ZPathEdit(cbSwitch, path, this);
    pLayout->addWidget(new QLabel("Set NASLOC"));
    pLayout->addWidget(m_pPthEdit);
    pLayout->addItem(new QSpacerItem(10, 10, QSizePolicy::Expanding));
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
}

void ZNASLOCPane::saveValue()
{
    auto& inst = ZenoSettingsManager::GetInstance();
    const QString& oldPath = inst.getValue(zsNASLOC).toString();
    QString path = m_pPthEdit->text();
    path.replace('\\', '/');
    if (inst.setValue(zsNASLOC, path))
    {
        startUp(true);
    }
}

//ZenoCache Pane
ZenoCachePane::ZenoCachePane(QWidget* parent) : QWidget(parent)
{
    auto& inst = ZenoSettingsManager::GetInstance();

    QVariant varEnableCache = inst.getValue(zsCacheEnable);
    QVariant varTempCacheDir = inst.getValue(zsCacheAutoRemove);
    QVariant varCacheRoot = inst.getValue(zsCacheDir);
    QVariant varCacheNum = inst.getValue(zsCacheNum);
    QVariant varAutoCleanCache = inst.getValue(zsCacheAutoClean);

    bool bEnableCache = varEnableCache.isValid() ? varEnableCache.toBool() : false;
    bool bTempCacheDir = varTempCacheDir.isValid() ? varTempCacheDir.toBool() : false;
    QString cacheRootDir = varCacheRoot.isValid() ? varCacheRoot.toString() : "";
    int cacheNum = varCacheNum.isValid() ? varCacheNum.toInt() : 1;
    bool bAutoCleanCache = varAutoCleanCache.isValid() ? varAutoCleanCache.toBool() : true;

    CALLBACK_SWITCH cbSwitch = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn); //deal with ubuntu dialog slow problem when update viewport.
    };
    m_pPathEdit = new ZPathEdit(cbSwitch, cacheRootDir);
    m_pPathEdit->setFixedWidth(256);
    m_pPathEdit->setEnabled(!bTempCacheDir && bEnableCache);
    m_pTempCacheDir = new QCheckBox;
    m_pTempCacheDir->setCheckState(bTempCacheDir ? Qt::Checked : Qt::Unchecked);
    m_pTempCacheDir->setEnabled(bEnableCache);
    m_pAutoCleanCache = new QCheckBox;
    m_pAutoCleanCache->setCheckState(bAutoCleanCache ? Qt::Checked : Qt::Unchecked);
    m_pAutoCleanCache->setEnabled(bEnableCache && !bTempCacheDir);

    connect(m_pTempCacheDir, &QCheckBox::stateChanged, [=](bool state) {
        m_pPathEdit->setText("");
        m_pPathEdit->setEnabled(!state);
        m_pAutoCleanCache->setChecked(Qt::Unchecked);
        m_pAutoCleanCache->setEnabled(!state);
    });

    m_pCacheNumSpinBox = new QSpinBox;
    m_pCacheNumSpinBox->setRange(1, 10000);
    m_pCacheNumSpinBox->setValue(cacheNum);
    m_pCacheNumSpinBox->setEnabled(bEnableCache);

    m_pEnableCheckbox = new QCheckBox;
    m_pEnableCheckbox->setCheckState(bEnableCache ? Qt::Checked : Qt::Unchecked);
    connect(m_pEnableCheckbox, &QCheckBox::stateChanged, [=](bool state) {
        if (!state)
        {
            m_pCacheNumSpinBox->clear();
            m_pPathEdit->clear();
            m_pTempCacheDir->setCheckState(Qt::Unchecked);
            m_pAutoCleanCache->setCheckState(Qt::Unchecked);
        }
        m_pCacheNumSpinBox->setEnabled(state);
        m_pPathEdit->setEnabled(state);
        m_pTempCacheDir->setEnabled(state);
        m_pAutoCleanCache->setEnabled(state && !m_pTempCacheDir->isChecked());
    });

    QGridLayout* pLayout = new QGridLayout(this);
    pLayout->addWidget(new QLabel(tr("Enable cache")), 0, 0);
    pLayout->addWidget(m_pEnableCheckbox, 0, 1);
    pLayout->addWidget(new QLabel(tr("Cache num")), 1, 0);
    pLayout->addWidget(m_pCacheNumSpinBox, 1, 1);
    pLayout->addWidget(new QLabel(tr("Temp cache directory")), 2, 0);
    pLayout->addWidget(m_pTempCacheDir, 2, 1);
    pLayout->addWidget(new QLabel(tr("Cache root")), 3, 0);
    pLayout->addWidget(m_pPathEdit, 3, 1);
    pLayout->addWidget(new QLabel(tr("Cache auto clean up")), 4, 0);
    pLayout->addWidget(m_pAutoCleanCache, 4, 1);
    QSpacerItem* pSpacerItem = new QSpacerItem(10, 10, QSizePolicy::Expanding);
    pLayout->addItem(pSpacerItem, 0, 2, 5);
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
}

void ZenoCachePane::saveValue()
{
    auto& inst = ZenoSettingsManager::GetInstance();
    inst.setValue(zsCacheEnable, m_pEnableCheckbox->checkState() == Qt::Checked);
    inst.setValue(zsCacheAutoRemove, m_pTempCacheDir->checkState() == Qt::Checked);
    inst.setValue(zsCacheDir, m_pPathEdit->text());
    inst.setValue(zsCacheNum, m_pCacheNumSpinBox->value());
    inst.setValue(zsCacheAutoClean, m_pAutoCleanCache->checkState() == Qt::Checked);
}

//Ray Tracing Pane
ZRayTracingPane::ZRayTracingPane(QWidget* parent) : QWidget(parent)
{
    auto& inst = ZenoSettingsManager::GetInstance();
    QVariant varEnableShiftChangeFOV = inst.getValue(zsEnableShiftChangeFOV);
    QVariant varViewportPointSizeScale = inst.getValue(zsViewportPointSizeScale);

    bool bEnableShiftChangeFOV = varEnableShiftChangeFOV.isValid() ? varEnableShiftChangeFOV.toBool() : true;
    double viewportPointSizeScale = varViewportPointSizeScale.isValid() ? varViewportPointSizeScale.toDouble() : 1;

    m_pEnableShiftChangeFOV = new QCheckBox;
    m_pEnableShiftChangeFOV->setCheckState(bEnableShiftChangeFOV ? Qt::Checked : Qt::Unchecked);

    m_pViewportPointSizeScaleSpinBox = new QDoubleSpinBox;
    m_pViewportPointSizeScaleSpinBox->setValue(viewportPointSizeScale);

    m_pViewportSampleNumber = new QSpinBox;
    m_pViewportSampleNumber->setRange(1, 10000);
    m_pViewportSampleNumber->setValue(1);
    auto main = zenoApp->getMainWindow();
    if (main) for (auto displaywid : main->viewports()) {
        if (displaywid && !displaywid->isGLViewport()) {
            if (auto vis = displaywid->getZenoVis()) {
                if (auto scene = vis->getSession()->get_scene()) {
                    m_pViewportSampleNumber->setValue(scene->drawOptions->num_samples);
                }
            }
        }
    }

    auto settingOr = [&inst](const char* name, const QVariant& fallback) {
        const QVariant value = inst.getValue(name);
        return value.isValid() ? value : fallback;
    };

    m_pDenoise = new QCheckBox;
    m_pDenoise->setChecked(settingOr(zsRayTracingDenoise, false).toBool());

    m_pMaxBounce = new QSpinBox;
    m_pMaxBounce->setRange(1, 64);
    m_pMaxBounce->setValue(settingOr(zsRayTracingMaxBounce, 4).toInt());

    m_pIntegrator = new QComboBox;
    m_pIntegrator->setObjectName("rayTracingIntegrator");
    m_pIntegrator->addItem(tr("Path Tracing"), "path");
    m_pIntegrator->addItem(tr("Stochastic Progressive Photon Mapping"), "sppm");
    m_pIntegrator->setCurrentIndex(std::max(0, m_pIntegrator->findData(
        settingOr(zsRayTracingIntegrator, "path"))));
    m_pIntegrator->setToolTip(tr("SPPM renders surface light transport and caustics. Unsupported scenes use Path Tracing and report the reason in the render log."));
    m_pSppmPhotons = new QSpinBox;
    m_pSppmPhotons->setRange(0, 1 << 28);
    m_pSppmPhotons->setSpecialValueText(tr("Automatic (one per pixel)"));
    m_pSppmPhotons->setValue(settingOr(zsSppmPhotons, 0).toInt());
    m_pSppmPhotons->setToolTip(tr("Photon emission attempts per iteration. Viewport Sample Number controls the number of iterations."));
    m_pSppmRadius = new QDoubleSpinBox;
    m_pSppmRadius->setObjectName("photonWorldRadius");
    m_pSppmRadius->setDecimals(6);
    m_pSppmRadius->setRange(0.0, 1000000.0);
    m_pSppmRadius->setSpecialValueText(tr("Automatic (scene size)"));
    m_pSppmRadius->setValue(settingOr(zsSppmRadius, 0.0).toDouble());
    m_pSppmRadius->setToolTip(tr("Initial gather radius in scene units. Smaller radii preserve detail but need more photons."));
    const double savedPixelRadius = settingOr(zsSppmRadiusPixels,
        inst.getValue(zsSppmRadius).isValid() ? 0.0 : 4.0).toDouble();
    m_pSppmRadiusMode = new QComboBox;
    m_pSppmRadiusMode->setObjectName("photonRadiusMode");
    m_pSppmRadiusMode->addItem(tr("Pixel size (camera scaled)"), "pixel");
    m_pSppmRadiusMode->addItem(tr("World units"), "world");
    m_pSppmRadiusMode->setCurrentIndex(savedPixelRadius > 0.0 ? 0 : 1);
    m_pSppmRadiusPixels = new QDoubleSpinBox;
    m_pSppmRadiusPixels->setObjectName("photonPixelRadius");
    m_pSppmRadiusPixels->setDecimals(2);
    m_pSppmRadiusPixels->setRange(0.01, 1000.0);
    m_pSppmRadiusPixels->setSingleStep(0.5);
    m_pSppmRadiusPixels->setSuffix(tr(" px"));
    m_pSppmRadiusPixels->setValue(savedPixelRadius > 0.0 ? savedPixelRadius : 4.0);
    m_pSppmRadiusPixels->setToolTip(tr("Converts pixel size to a shared initial world radius at the nearest camera-sampled caster (or nearest sampled surface). Recalculated when the camera or resolution changes. This is not a separate depth-adaptive radius at every pixel."));
    m_pSppmAlpha = new QDoubleSpinBox;
    m_pSppmAlpha->setDecimals(6);
    m_pSppmAlpha->setRange(0.01, 1.0);
    m_pSppmAlpha->setSingleStep(0.05);
    m_pSppmAlpha->setValue(settingOr(zsSppmAlpha, 2.0 / 3.0).toDouble());
    m_pSppmAlpha->setToolTip(tr("Progressive radius reduction parameter. The standard value is 2/3."));
    m_pSppmGuiding = new QCheckBox;
    m_pSppmGuiding->setObjectName("photonGuiding");
    m_pSppmGuiding->setChecked(settingOr(zsSppmGuiding, true).toBool());
    m_pSppmSearch = new QComboBox;
    m_pSppmSearch->addItem(tr("Spatial hash grid"), "hash");
    m_pSppmSearch->addItem(tr("Hardware ray tracing (BVH)"), "bvh");
    m_pSppmSearch->setCurrentIndex(qMax(0, m_pSppmSearch->findData(settingOr(zsSppmSearch, "hash"))));
    m_pSppmSearch->setToolTip(tr("Choose how photons find nearby receivers. Both methods use the same radius and lighting estimator. BVH build cost is included in rendering time."));
    m_pSppmGuiding->setToolTip(tr("Learn which lights and emission directions contribute to the photon image, including caustics. Retains uniform exploration and full photon energy. May add overhead in simple scenes."));

    m_pEnableSharc = new QCheckBox;
    m_pEnableSharc->setObjectName("enableSharc");
    m_pEnableSharc->setChecked(settingOr(zsEnableSharc, true).toBool());
    m_pEnableSharc->setToolTip(tr("Used by Path Tracing, including unsupported-scene fallback. Photon rendering bypasses SHARC. You can turn this preference off in either integrator."));

    m_pSharcResetCache = new QPushButton(tr("Reset"), this);
    m_pSharcResetCache->setAutoDefault(false);
    m_pSharcResetCache->setToolTip(tr("Reset the active SHARC cache immediately"));
    connect(m_pSharcResetCache, &QPushButton::clicked, this, []() {
        auto main = zenoApp->getMainWindow();
        if (!main) {
            return;
        }
        for (auto displaywid : main->viewports()) {
            if (displaywid) {
                displaywid->resetSharcCache();
            }
        }
    });

    m_pSharcDownscaleFactor = new QSpinBox;
    m_pSharcDownscaleFactor->setRange(1, 10);
    m_pSharcDownscaleFactor->setValue(settingOr(zsSharcDownscaleFactor, 5).toInt());

    m_pSharcSceneScale = new QDoubleSpinBox;
    m_pSharcSceneScale->setDecimals(4);
    m_pSharcSceneScale->setRange(0.0001, 1000000.0);
    m_pSharcSceneScale->setValue(settingOr(zsSharcSceneScale, 50.0).toDouble());

    m_pSharcAccumulationFrameNum = new QSpinBox;
    m_pSharcAccumulationFrameNum->setRange(1, 1024);
    m_pSharcAccumulationFrameNum->setValue(
        settingOr(zsSharcAccumulationFrameNum, 20).toInt());

    m_pSharcStaleFrameNum = new QSpinBox;
    m_pSharcStaleFrameNum->setRange(1, 1024);
    m_pSharcStaleFrameNum->setValue(settingOr(zsSharcStaleFrameNum, 60).toInt());

    m_pSharcRoughnessMin = new QDoubleSpinBox;
    m_pSharcRoughnessMin->setDecimals(3);
    m_pSharcRoughnessMin->setSingleStep(0.05);
    m_pSharcRoughnessMin->setRange(0.0, 1.0);
    m_pSharcRoughnessMin->setValue(settingOr(zsSharcRoughnessMin, 0.4).toDouble());

    m_pSharcDebugMode = new QCheckBox;
    m_pSharcDebugMode->setChecked(settingOr(zsSharcDebugMode, false).toBool());

    m_pSharcMaterialDemodulation = new QCheckBox;
    m_pSharcMaterialDemodulation->setChecked(
        settingOr(zsSharcMaterialDemodulation, false).toBool());
    m_pSharcMaterialDemodulation->setToolTip(
        tr("Experimental: requires a material response that exactly matches the active BSDF"));

    m_pSharcSHEncoding = new QCheckBox;
    m_pSharcSHEncoding->setChecked(settingOr(zsSharcSHEncoding, false).toBool());
    m_pSharcSHEncoding->setToolTip(
        tr("Preserve directional glossy/specular radiance in SHARC; increases cache memory and update cost"));

    auto layout = new QVBoxLayout(this);
    layout->setAlignment(Qt::AlignTop);
    auto common = new QFormLayout;
    common->addRow(tr("Integrator"), m_pIntegrator);
    common->addRow(tr("Viewport Sample Number"), m_pViewportSampleNumber);
    common->addRow(tr("Denoise"), m_pDenoise);
    common->addRow(tr("Max Bounce"), m_pMaxBounce);
    common->addRow(tr("Enable SHARC for Path Tracing"), m_pEnableSharc);
    auto sharcStatus = new QLabel;
    sharcStatus->setObjectName("sharcStatus");
    sharcStatus->setWordWrap(true);
    common->addRow(sharcStatus);
    layout->addLayout(common);

    auto photons = new QGroupBox(tr("Photon Mapping"));
    auto photonLayout = new QFormLayout(photons);
    photonLayout->addRow(tr("Gather radius mode"), m_pSppmRadiusMode);
    photonLayout->addRow(tr("Initial radius in pixels"), m_pSppmRadiusPixels);
    photonLayout->addRow(tr("Initial radius in world units"), m_pSppmRadius);
    auto radiusStatus = new QLabel;
    radiusStatus->setObjectName("photonRadiusStatus");
    radiusStatus->setWordWrap(true);
    photonLayout->addRow(radiusStatus);
    photonLayout->addRow(tr("Photons per iteration"), m_pSppmPhotons);
    photonLayout->addRow(tr("Progressive radius alpha"), m_pSppmAlpha);
    photonLayout->addRow(tr("Photon guiding"), m_pSppmGuiding);
    auto guidingStatus = new QLabel;
    guidingStatus->setObjectName("photonGuidingStatus");
    guidingStatus->setWordWrap(true);
    photonLayout->addRow(guidingStatus);
    photonLayout->addRow(tr("Photon search"), m_pSppmSearch);
    layout->addWidget(photons);

    auto sharc = new QGroupBox(tr("SHARC Settings"));
    sharc->setObjectName("sharcSettings");
    auto sharcLayout = new QFormLayout(sharc);
    sharcLayout->addRow(tr("Reset cache"), m_pSharcResetCache);
    sharcLayout->addRow(tr("Downscale factor"), m_pSharcDownscaleFactor);
    sharcLayout->addRow(tr("Scene scale"), m_pSharcSceneScale);
    sharcLayout->addRow(tr("Accumulation frame number"), m_pSharcAccumulationFrameNum);
    sharcLayout->addRow(tr("Stale frame number"), m_pSharcStaleFrameNum);
    sharcLayout->addRow(tr("Minimum roughness"), m_pSharcRoughnessMin);
    sharcLayout->addRow(tr("Debug mode"), m_pSharcDebugMode);
    sharcLayout->addRow(tr("Material demodulation"), m_pSharcMaterialDemodulation);
    sharcLayout->addRow(tr("Directional SH encoding"), m_pSharcSHEncoding);
    layout->addWidget(sharc);

    auto viewport = new QGroupBox(tr("Viewport Controls"));
    auto viewportLayout = new QFormLayout(viewport);
    viewportLayout->addRow(tr("Enable Shift change FOV"), m_pEnableShiftChangeFOV);
    viewportLayout->addRow(tr("Point size scale"), m_pViewportPointSizeScaleSpinBox);
    layout->addWidget(viewport);

    auto updateIntegratorControls = [this, photons, sharc, sharcStatus, radiusStatus, guidingStatus]() {
        const bool sppm = m_pIntegrator->currentData().toString() == "sppm";
        const bool pixels = m_pSppmRadiusMode->currentData().toString() == "pixel";
        photons->setEnabled(sppm);
        m_pSppmRadius->setEnabled(!pixels);
        m_pSppmRadiusPixels->setEnabled(pixels);
        guidingStatus->setText(m_pSppmGuiding->isChecked()
            ? tr("Photon guiding is enabled for progressive SPPM.") : tr("Photon guiding is off."));
        radiusStatus->setText(tr("SPPM progressively shrinks the gather radius at each receiver. Pixel mode scales the shared starting radius with the camera."));
        // Keep the preference editable even while photon transport bypasses it.
        sharc->setVisible(!sppm && m_pEnableSharc->isChecked());
        sharcStatus->setText(sppm
            ? (m_pEnableSharc->isChecked()
                ? tr("SHARC is bypassed for photons; enabled only if the scene falls back to Path Tracing.")
                : tr("SHARC is off, including Path Tracing fallback."))
            : (m_pEnableSharc->isChecked() ? tr("SHARC is enabled for Path Tracing.") : tr("SHARC is off.")));
    };
    connect(m_pIntegrator, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [updateIntegratorControls](int) { updateIntegratorControls(); });
    connect(m_pSppmRadiusMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [updateIntegratorControls](int) { updateIntegratorControls(); });
    connect(m_pSppmGuiding, &QCheckBox::toggled, this, [updateIntegratorControls](bool) { updateIntegratorControls(); });
    connect(m_pEnableSharc, &QCheckBox::toggled, this, [updateIntegratorControls](bool) { updateIntegratorControls(); });
    updateIntegratorControls();
}

void ZRayTracingPane::saveValue()
{
    auto& inst = ZenoSettingsManager::GetInstance();
    const QVariant oldSceneScale = inst.getValue(zsSharcSceneScale);
    const QVariant oldMaterialDemodulation = inst.getValue(zsSharcMaterialDemodulation);
    const QVariant oldSHEncoding = inst.getValue(zsSharcSHEncoding);
    const bool resetSharc =
        (oldSceneScale.isValid() &&
         oldSceneScale.toDouble() != m_pSharcSceneScale->value()) ||
        (oldMaterialDemodulation.isValid() &&
         oldMaterialDemodulation.toBool() != m_pSharcMaterialDemodulation->isChecked()) ||
        (oldSHEncoding.isValid() ? oldSHEncoding.toBool() : false) !=
            m_pSharcSHEncoding->isChecked();

    inst.setValue(zsEnableShiftChangeFOV, m_pEnableShiftChangeFOV->checkState() == Qt::Checked);
    inst.setValue(zsViewportPointSizeScale, m_pViewportPointSizeScaleSpinBox->value());
    inst.setValue(zsRayTracingDenoise, m_pDenoise->isChecked());
    inst.setValue(zsRayTracingMaxBounce, m_pMaxBounce->value());
    inst.setValue(zsRayTracingIntegrator, m_pIntegrator->currentData());
    inst.setValue(zsSppmPhotons, m_pSppmPhotons->value());
    inst.setValue(zsSppmRadius, m_pSppmRadius->value());
    const double radiusPixels = m_pSppmRadiusMode->currentData().toString() == "pixel"
        ? m_pSppmRadiusPixels->value() : 0.0;
    inst.setValue(zsSppmRadiusPixels, radiusPixels);
    inst.setValue(zsSppmGuiding, m_pSppmGuiding->isChecked());
    inst.setValue(zsSppmSearch, m_pSppmSearch->currentData());
    inst.setValue(zsSppmAlpha, m_pSppmAlpha->value());
    inst.setValue(zsEnableSharc, m_pEnableSharc->isChecked());
    inst.setValue(zsSharcDownscaleFactor, m_pSharcDownscaleFactor->value());
    inst.setValue(zsSharcSceneScale, m_pSharcSceneScale->value());
    inst.setValue(zsSharcAccumulationFrameNum, m_pSharcAccumulationFrameNum->value());
    inst.setValue(zsSharcStaleFrameNum, m_pSharcStaleFrameNum->value());
    inst.setValue(zsSharcRoughnessMin, m_pSharcRoughnessMin->value());
    inst.setValue(zsSharcDebugMode, m_pSharcDebugMode->isChecked());
    inst.setValue(zsSharcMaterialDemodulation, m_pSharcMaterialDemodulation->isChecked());
    inst.setValue(zsSharcSHEncoding, m_pSharcSHEncoding->isChecked());

    auto main = zenoApp->getMainWindow();
    if (!main) return;
    for (auto displaywid : main->viewports()) {
        if (displaywid) {
            displaywid->setSampleNumber(m_pViewportSampleNumber->value());
            displaywid->setSppmSettings(m_pIntegrator->currentData().toString(),
                m_pSppmPhotons->value(), static_cast<float>(m_pSppmRadius->value()),
                static_cast<float>(m_pSppmAlpha->value()), static_cast<float>(radiusPixels), m_pSppmGuiding->isChecked(), m_pSppmSearch->currentData().toString());
            displaywid->setRayTracingSettings(
                m_pDenoise->isChecked(),
                m_pMaxBounce->value(),
                m_pEnableSharc->isChecked(),
                resetSharc,
                m_pSharcDownscaleFactor->value(),
                static_cast<float>(m_pSharcSceneScale->value()),
                m_pSharcAccumulationFrameNum->value(),
                m_pSharcStaleFrameNum->value(),
                static_cast<float>(m_pSharcRoughnessMin->value()),
                m_pSharcDebugMode->isChecked(),
                m_pSharcMaterialDemodulation->isChecked(),
                m_pSharcSHEncoding->isChecked());
        }
    }
}

//layout pane
ZLayoutPane::ZLayoutPane(QWidget* parent) : QWidget(parent)
{
    QHBoxLayout* pLayout = new QHBoxLayout(this);
    m_listWidget = new QListWidget(this);
    m_listWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    pLayout->addWidget(m_listWidget);
    pLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);

    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("layout");
    QStringList lst = settings.childGroups();
    if (lst.contains("LatestLayout"))
        lst.removeOne("LatestLayout");
    if (lst.isEmpty())
        return;
    m_listWidget->addItems(lst);
    for (int i = 0; i < m_listWidget->count(); i++)
    {
        if (QListWidgetItem* pItem = m_listWidget->item(i))
            pItem->setData(Qt::UserRole, pItem->text());
    }

    QPushButton* deleteButton = new QPushButton(tr("Delete"), this);
    deleteButton->setEnabled(false);
    QPushButton* renameButton = new QPushButton(tr("Rename"), this);
    renameButton->setEnabled(false);

    QVBoxLayout *pVLayout = new QVBoxLayout;
    pVLayout->addWidget(renameButton);
    pVLayout->addWidget(deleteButton);
    pVLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    pLayout->addLayout(pVLayout);

    connect(m_listWidget, &QListWidget::itemSelectionChanged, this, [=]() {
        int count = m_listWidget->selectedItems().size();
        deleteButton->setEnabled(count > 0);
        renameButton->setEnabled(count == 1);
    });
    connect(deleteButton, &QPushButton::clicked, this, [=]() {
        QList<QListWidgetItem*> lst = m_listWidget->selectedItems();
        for (const auto& pItem : lst)
        {
            QString key = pItem->data(Qt::UserRole).toString();
            int row = m_listWidget->row(pItem);
            m_listWidget->takeItem(row);
            const char* prop = "delItems";
            QStringList lst = m_listWidget->property(prop).toStringList();
            lst << key;
            m_listWidget->setProperty(prop, lst);
        }
    });

    connect(renameButton, &QPushButton::clicked, this, [=]() {
        QListWidgetItem* pItem = m_listWidget->currentItem();
        if (!pItem)
            return;
        const QString& key = pItem->data(Qt::DisplayRole).toString();
        QString newkey = QInputDialog::getText(this, tr("Rename"), tr("Name"), QLineEdit::Normal, key);
        if (newkey.isEmpty())
            return;
        pItem->setData(Qt::DisplayRole, newkey);
    });
}

void ZLayoutPane::saveValue()
{
    bool bChanged = false;
    //delete layout
    const char* prop = "delItems";
    QStringList lst = m_listWidget->property(prop).toStringList();
    if (!lst.isEmpty())
    {
        bChanged = true;
        for (const auto& delKey : lst)
        {
            QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
            settings.beginGroup("layout");
            settings.remove(delKey);
            settings.endGroup();
        }
    }

    //rename layout
    for (int i = 0; i < m_listWidget->count(); i++)
    {
        QListWidgetItem* pItem = m_listWidget->item(i);
        const QString& newkey = pItem->data(Qt::DisplayRole).toString();
        const QString& key = pItem->data(Qt::UserRole).toString();
        if (key != newkey)
        {
            QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
            settings.beginGroup("layout");
            settings.beginGroup(key);
            QVariant value = settings.value("content");
            settings.endGroup();
            settings.remove(key);
            settings.beginGroup(newkey);
            settings.setValue("content", value);
            settings.endGroup();
            settings.endGroup();
            bChanged = true;
        }
    }
    if (bChanged)
    {
        zenoApp->getMainWindow()->loadSavedLayout();
    }
}

//TabBar
QSize ZPreferencesTabBar::tabSizeHint(int index) const {
    QSize s = QTabBar::tabSizeHint(index);
    s.transpose();
    return s;
}

void ZPreferencesTabBar::paintEvent(QPaintEvent* event) {
    QStylePainter painter(this);
    QStyleOptionTab opt;

    for (int i = 0; i < count(); i++)
    {
        initStyleOption(&opt, i);
        painter.drawControl(QStyle::CE_TabBarTabShape, opt);
        painter.save();

        QSize s = opt.rect.size();
        s.transpose();
        QRect r(QPoint(), s);
        r.moveCenter(opt.rect.center());
        opt.rect = r;

        QPoint c = tabRect(i).center();
        painter.translate(c);
        painter.rotate(90);
        painter.translate(-c);
        painter.drawControl(QStyle::CE_TabBarTabLabel, opt);
        painter.restore();
    }
}

ZPreferencesTabWidget::ZPreferencesTabWidget(QWidget* parent)
{

    initUI();
}

void ZPreferencesTabWidget::initUI()
{
    setTabBar(new ZPreferencesTabBar());
    setTabPosition(QTabWidget::West);
    setProperty("cssClass", "preferences");
    //language
    m_pLanguagePane = new ZLanguagePane(this);
    addTab(m_pLanguagePane, tr("Language"));
    //NASLOC pane
    m_pNASLOCPane = new ZNASLOCPane(this);
    addTab(m_pNASLOCPane, tr("NASLOC"));
    //ZenoCache pane
    m_pZenoCachePane = new ZenoCachePane(this);
    addTab(m_pZenoCachePane, tr("Zeno Cache"));
    //ray tracing pane
    m_pRayTracingPane = new ZRayTracingPane(this);
    auto rayTracingScroll = new QScrollArea(this);
    rayTracingScroll->setObjectName("rayTracingScroll");
    rayTracingScroll->setWidgetResizable(true);
    rayTracingScroll->setFrameShape(QFrame::NoFrame);
    rayTracingScroll->setWidget(m_pRayTracingPane);
    // QScrollArea enables a light autofill background on its content by default.
    // Keep the preferences dialog's dark background through both layers.
    rayTracingScroll->viewport()->setAutoFillBackground(false);
    m_pRayTracingPane->setAutoFillBackground(false);
    addTab(rayTracingScroll, tr("Ray Tracing"));
    //shortcut panel
    m_pShortcutsPane = new ShortcutsPane(this);
    addTab(m_pShortcutsPane, tr("Shortcuts"));
    //layout manage
    m_pLayoutPane = new ZLayoutPane(this);
    addTab(m_pLayoutPane, tr("Layout Manage"));
}

void ZPreferencesTabWidget::saveSettings()
{
    //language
    m_pLanguagePane->saveValue();
    //NASLOC
    m_pNASLOCPane->saveValue();
    //zeno cache
    m_pZenoCachePane->saveValue();
    //ray tracing
    m_pRayTracingPane->saveValue();
    //shortcuts
    m_pShortcutsPane->saveValue();
    //layout
    m_pLayoutPane->saveValue();
}

ZPreferencesDlg::ZPreferencesDlg(QWidget* parent) : ZFramelessDialog(parent), m_pTabWidget(nullptr)
{
    initUI();
}

ZPreferencesDlg::~ZPreferencesDlg()
{
}

void ZPreferencesDlg::initUI()
{
    QWidget* pMainWidget = new QWidget(this);
    m_pTabWidget = new ZPreferencesTabWidget(this);
    m_pTabWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pOKBtn = new QPushButton(tr("OK"), this);
    m_pCancelBtn = new QPushButton(tr("Cancel"), this);
    QHBoxLayout* pHLyout = new QHBoxLayout();
    pHLyout->addWidget(m_pOKBtn);
    pHLyout->addWidget(m_pCancelBtn);
    pHLyout->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    QVBoxLayout* pLyout = new QVBoxLayout(pMainWidget);
    int margin = ZenoStyle::dpiScaled(10);
    pLyout->setContentsMargins(ZenoStyle::dpiScaled(2), margin, margin, margin);
    pLyout->addWidget(m_pTabWidget);
    pLyout->addLayout(pHLyout);
    pMainWidget->setLayout(pLyout);
    setTitleText(tr("Preferences"));
    setMainWidget(pMainWidget);
    resize(ZenoStyle::dpiScaledSize(QSize(800, 800)));
    setTitleIcon(QIcon(":/icons/toolbar_localSetting_idle.svg"));
    connect(m_pOKBtn, &QPushButton::clicked, this, [=]() {
        m_pTabWidget->saveSettings();
        accept();
    });
    connect(m_pCancelBtn, &QPushButton::clicked, this, &ZPreferencesDlg::reject);
}
