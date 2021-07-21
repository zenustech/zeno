from . import *

CURR_VERSION = 'v1'
MAX_STACK_LENGTH = 100

style = {
    'title_color': '#1e1e1e',
    'socket_connect_color': '#DDDDDD',
    'socket_unconnect_color': '#4a4a4a',
    'title_text_color': '#787878',
    'title_text_size': 14,
    'button_text_size': 10,
    'socket_text_size': 14,
    'param_text_size': 14,
    'socket_text_color': '#787878',
    'panel_color': '#282828',
    'blackboard_title_color': '#393939',
    'blackboard_panel_color': '#1B1B1B',
    'line_color': '#B0B0B0',
    'background_color': '#2C2C2C',
    'selected_color': '#EE8844',
    'button_color': '#1e1e1e',
    'button_text_color': '#ffffff',
    'button_selected_color': '#449922',
    'button_selected_text_color': '#333333',
    'output_shift': 1,

    'line_width': 3,
    'node_outline_width': 2,
    'socket_outline_width': 2,
    'node_rounded_radius': 6,
    'socket_radius': 8,
    'node_width': 240,
    'text_height': 23,
    'copy_offset_x': 100,
    'copy_offset_y': 100,
    'hori_margin': 9,
    'dummy_socket_offset': 15,
}

TEXT_HEIGHT = style['text_height']
HORI_MARGIN = style['hori_margin']
SOCKET_RADIUS = style['socket_radius']
BEZIER_FACTOR = 0.5

def fillRect(painter, rect, color, line_width=None, line_color=None):
    if line_width:
        painter.fillRect(rect, QColor(line_color))

        w = line_width
        r = rect
        content_rect = QRect(r.x() + w, r.y() + w, r.width() - w * 2, r.height() - w * 2)
        painter.fillRect(content_rect, QColor(color))
    else:
        painter.fillRect(rect, QColor(color))

def fillRectOld(painter, rect, color, line_width=None, line_color=None):
    if line_width:
        pen = QPen(QColor(line_color))
        pen.setWidth(line_width)
        pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(pen)
    else:
        painter.setPen(Qt.NoPen)

    painter.setBrush(QColor(color))
    pathTitle = QPainterPath()
    pathTitle.addRect(rect)
    painter.drawPath(pathTitle.simplified())

from . import *
