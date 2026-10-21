"""Graph 自动布局（手册 §10.6 自动布局 / §33.29.1）。

自动布局不是 UI 美化，而是 Scene Build 后 Graph 可维护性的正式步骤。
第一阶段算法归 Python 公共层，复用现有命令通道：

    graphGeometry → 读取 node position/size/bbox/topology
    → layout.py 生成 [{node_id, x, y}] → setNodePosList → graphGeometry 二次读取
    → Collision / Overlap Check

算法：
    1. 从 link topology 构建依赖 DAG；
    2. 拓扑层决定 left-to-right 层级；
    3. 用 node width/height 计算层内间距；
    4. 批量提交位置；
    5. 二次读取 graphGeometry 验证；
    6. 重叠超阈值则增大 layer/row gap 重算；
    7. 仍失败才返回 LAYOUT_FAILED。

本模块的 `compute_layout` 是**纯函数**（输入拓扑 + 尺寸，输出坐标与碰撞报告），
读取 graphGeometry / 提交坐标由 capability 层编排，便于离线单测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

CODE_LAYOUT_FAILED = "layout_failed"

DEFAULT_NODE_SIZE = (240.0, 120.0)
DEFAULT_LAYER_GAP = 320.0
DEFAULT_ROW_GAP = 160.0


@dataclass
class LayoutResult:
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    layers: dict[str, int] = field(default_factory=dict)
    collisions: list[tuple[str, str]] = field(default_factory=list)
    attempts: int = 0
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def compute_layout(
    node_ids: list[str],
    edges: list[tuple[str, str]],
    sizes: Optional[dict[str, tuple[float, float]]] = None,
    *,
    layer_gap: float = DEFAULT_LAYER_GAP,
    row_gap: float = DEFAULT_ROW_GAP,
    max_attempts: int = 5,
) -> LayoutResult:
    """拓扑分层 + left-to-right 布局 + 碰撞检查（重叠则增大间距重算）。"""
    ids = [str(n) for n in node_ids]
    if not ids:
        return LayoutResult(status="ok")
    size_of = _size_lookup(ids, sizes)

    layers = _topological_layers(ids, edges)
    attempts = 0
    gap_l, gap_r = layer_gap, row_gap
    result: Optional[LayoutResult] = None
    while attempts < max_attempts:
        attempts += 1
        positions = _place(ids, layers, size_of, gap_l, gap_r)
        collisions = _find_collisions(ids, positions, size_of)
        result = LayoutResult(
            positions=positions, layers=layers, collisions=collisions, attempts=attempts
        )
        if not collisions:
            return result
        gap_l *= 1.4
        gap_r *= 1.4
    assert result is not None
    result.status = CODE_LAYOUT_FAILED
    return result


def build_batch_items(
    positions: dict[str, tuple[float, float]],
    alias_to_uuid: dict[str, str],
) -> str:
    """把坐标映射转成 setNodePosList 的 items 串（"uuid:x,y;..."）。"""
    parts: list[str] = []
    for alias, (x, y) in positions.items():
        uuid = alias_to_uuid.get(alias)
        if not uuid:
            continue
        parts.append(f"{uuid}:{_num(x)},{_num(y)}")
    return ";".join(parts)


# --- 内部 ---

def _size_lookup(
    ids: list[str], sizes: Optional[dict[str, tuple[float, float]]]
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for node_id in ids:
        raw = (sizes or {}).get(node_id)
        if raw and len(raw) >= 2 and raw[0] and raw[1]:
            out[node_id] = (float(raw[0]), float(raw[1]))
        else:
            out[node_id] = DEFAULT_NODE_SIZE
    return out


def _topological_layers(ids: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """按最长路径分层：源节点 layer=0，下游节点 = 上游最大 layer + 1。"""
    present = set(ids)
    preds: dict[str, set[str]] = {n: set() for n in ids}
    for out_id, in_id in edges:
        if out_id in present and in_id in present and out_id != in_id:
            preds[in_id].add(out_id)

    layers: dict[str, int] = {}
    for node in _topo_order(ids, preds):
        layer = 0
        for pred in preds[node]:
            layer = max(layer, layers.get(pred, 0) + 1)
        layers[node] = layer
    return layers


def _topo_order(ids: list[str], preds: dict[str, set[str]]) -> list[str]:
    """Kahn 拓扑排序；存在环时把剩余节点按原顺序追加（保证不丢失节点）。"""
    order: list[str] = []
    indeg = {n: len(preds[n]) for n in ids}
    ready = [n for n in ids if indeg[n] == 0]
    while ready:
        node = ready.pop(0)
        order.append(node)
        for other in ids:
            if node in preds[other]:
                indeg[other] -= 1
                if indeg[other] == 0:
                    ready.append(other)
    if len(order) < len(ids):
        order.extend(n for n in ids if n not in order)
    return order


def _place(
    ids: list[str],
    layers: dict[str, int],
    sizes: dict[str, tuple[float, float]],
    layer_gap: float,
    row_gap: float,
) -> dict[str, tuple[float, float]]:
    """按层分列、层内纵向堆叠。"""
    by_layer: dict[int, list[str]] = {}
    for node in ids:
        by_layer.setdefault(layers.get(node, 0), []).append(node)

    col_width: dict[int, float] = {}
    for layer, members in by_layer.items():
        col_width[layer] = max(sizes[m][0] for m in members)

    x_of: dict[int, float] = {}
    cursor = 0.0
    for layer in sorted(by_layer):
        x_of[layer] = cursor
        cursor += col_width[layer] + layer_gap

    positions: dict[str, tuple[float, float]] = {}
    for layer, members in by_layer.items():
        y = 0.0
        for node in members:
            positions[node] = (x_of[layer], y)
            y += sizes[node][1] + row_gap
    return positions


def _find_collisions(
    ids: list[str],
    positions: dict[str, tuple[float, float]],
    sizes: dict[str, tuple[float, float]],
) -> list[tuple[str, str]]:
    """两两包围盒重叠检查。"""
    boxes: list[tuple[str, float, float, float, float]] = []
    for node in ids:
        x, y = positions.get(node, (0.0, 0.0))
        w, h = sizes[node]
        boxes.append((node, x, y, x + w, y + h))
    collisions: list[tuple[str, str]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                collisions.append((a[0], b[0]))
    return collisions


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def geometry_sizes(raw: Any) -> dict[str, tuple[float, float]]:
    """把 graphGeometry 返回的 {nodes:[{uuid,width,height,...}]} 转成尺寸映射。"""
    sizes: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return sizes
    for node in raw.get("nodes") or []:
        uuid = str(node.get("uuid") or "")
        width = node.get("width")
        height = node.get("height")
        if uuid and width and height:
            sizes[uuid] = (float(width), float(height))
    return sizes
