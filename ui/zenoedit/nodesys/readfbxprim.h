#ifndef __READ_FBX_PRIM_H__
#define __READ_FBX_PRIM_H__

#include "zenonode.h"

class ReadFBXPrim : public ZenoNode
{
    Q_OBJECT
public:
    ReadFBXPrim(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~ReadFBXPrim();

    bool GenMode;

    void GenerateFBX();
protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onNodeClicked();
    void onPartClicked();
};

class NewFBXImportSkin : public ZenoNode
{
    Q_OBJECT
public:
    NewFBXImportSkin(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~NewFBXImportSkin();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onNodeClicked();
    void onPartClicked();
};

class EvalBlenderFile : public ZenoNode
{
    Q_OBJECT
  public:
    EvalBlenderFile(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~EvalBlenderFile();
    std::vector<std::string> GetCurNodeInputs();
  protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

  private slots:
    void onExecClicked();
    void onEvalClicked();
};

#endif