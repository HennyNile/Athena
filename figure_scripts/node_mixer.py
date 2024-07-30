#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pytikz import *
import math

class Concat(tikz_shape):
    def __init__(self, context: tikz_context):
        super().__init__(context)
        self.draw((0, 0), 'circle', '({0.5})')
        self.draw((-0.5, 0), '--', (0.5, 0))
        self.draw((0, -0.5), '--', (0, 0.5))

    def _anchors(self, x: float, y: float, scale: float):
        l = 0.25 * math.sqrt(2)
        return {
            'center': (x, y),
            'north': (x, y + 0.5 * scale),
            'south': (x, y - 0.5 * scale),
            'east': (x + 0.5 * scale, y),
            'west': (x - 0.5 * scale, y),
            'north east': (x + l * scale, y + l * scale),
            'north west': (x - l * scale, y + l * scale),
            'south east': (x + l * scale, y - l * scale),
            'south west': (x - l * scale, y - l * scale),
        }
    
    def _custom_anchor(self, degrees: float, x: float, y: float, scale: float):
        rad = math.radians(degrees)
        return (x + 0.5 * math.cos(rad) * scale, y + 0.5 * math.sin(rad) * scale)

class Rect(tikz_shape):
    def __init__(self, context: tikz_context, width: float, height: float, text: str = None, anchor: str = 'center', color: str = 'white', fill=False):
        super().__init__(context)
        self.width = width
        self.height = height
        if fill:
            self.fill((0, 0), 'rectangle', (width, height)).property(color)
        self.draw((0, 0), 'rectangle', (width, height))
        if text is not None:
            pos = (width / 2, height / 2)
            text_width = f'{{{width}}}cm'
            self.node('at', pos, f'[text width={text_width},align=center]', f'{{{{{text}}}}}').property(anchor=anchor)

    def _anchors(self):
        return {
            'center': (self.width / 2, self.height / 2),
            'north': (self.width / 2, self.height),
            'south': (self.width / 2, 0),
            'east': (self.width, self.height / 2),
            'west': (0, self.height / 2),
            'north east': (self.width, self.height),
            'north west': (0, self.height),
            'south east': (self.width, 0),
            'south west': (0, 0),
        }

class Blend(tikz_shape):
    def __init__(self, context: tikz_context):
        super().__init__(context)
        self.draw((0, 0), 'circle', '({0.5})')
        self.draw((0, -0.5), 'arc', '(-90:90:{0.45})')
        self.draw((0, 0.4), 'arc', '(90:270:{0.2})')
        self.draw((0, 0), 'arc', '(90:-90:{0.2})')
        self.draw((0, -0.4), 'arc', '(270:90:{0.45})')

    def _anchors(self) -> dict[str, tuple[float, float]]:
        return {
            'north': (0, 0.5),
            'south': (0, -0.5),
            'east': (0.5, 0),
            'west': (-0.5, 0),
        }

    def _custom_anchor(self, degrees: float) -> tuple[float, float]:
        rad = math.radians(degrees)
        return (math.cos(rad) * 0.5, math.sin(rad) * 0.5)

if __name__ == '__main__':
    context = tikz_context()
    context.use('arrows.meta')
    # context.add_color('inputcolor', '#F5CD53')
    # context.add_color('modulecolor', '#92E285')
    # context.add_color('datacolor', '#A6C4E8')
    # context.add_color('outputcolor', '#67ADE0')
    # context.add_color('labelcolor', '#F09E6F')
    # context.add_color('traincolor', '#6DBEBF')
    x = 0
    y = 0
    node_width = 0.75
    node_height = 0.6
    hpad = 0.5
    vpad = 0.4
    max_x = 0
    max_y = 0
    def draw_node(i, j, text):
        global max_x, max_y
        x_step = (node_width + hpad) / 2
        y_step = node_height + vpad
        right = x + j * x_step + node_width
        top = y + i * y_step + node_height
        if right > max_x:
            max_x = right
        if top > max_y:
            max_y = top
        return Rect(context, node_width, node_height, text).at(x + j * x_step, y + i * y_step).draw()
    nodes = {
        1: draw_node(3, 2, '1'),
        2: draw_node(2, 1, '2'),
        3: draw_node(1, 1, '3'),
        4: draw_node(0, 0, '4'),
        5: draw_node(0, 2, '5'),
        6: draw_node(2, 3, '6'),
    }
    def connect(node1, path1, node2, path2):
        context.draw(node1.anchor(path1), '--', node2.anchor(path2)).property('-{Stealth[length=5pt,width=3pt]}')
    connect(nodes[1], 'south', nodes[2], 'north')
    connect(nodes[2], 'south', nodes[3], 'north')
    connect(nodes[3], 'south', nodes[4], 'north')
    connect(nodes[3], 'south', nodes[5], 'north')
    connect(nodes[1], 'south', nodes[6], 'north')

    x = max_x + hpad / 2
    y = 0

    vec_height = 0.38
    print(max_y)
    assert max_y >= 8 * vec_height
    vpad = (max_y - 8 * vec_height) / 2
    result = Rect(context, 2, vec_height, 'Tree Repr').at(x, y).draw()
    y += vec_height + vpad
    max_pool = Rect(context, 3, vec_height, 'Max Pool').at(x - 0.5, y).draw()
    y += vec_height + vpad
    conv_nodes = {}
    for i, text in zip(range(6), ('6', '5', '4', '3,4,5', '2,3', '1,2,6')):
        conv_nodes[6-i] = Rect(context, 2, vec_height, f'{text}').at(x, y).draw()
        y += vec_height
    connect(conv_nodes[6], 'south', max_pool, 'north')
    connect(max_pool, 'south', result, 'north')

    bottomleft = nodes[2].anchor('south west')
    width = nodes[6].anchor('east')[0] - bottomleft[0]
    height = nodes[1].anchor('north')[1] - bottomleft[1]
    conv1 = Rect(context, width + 0.2, height + 0.12).at(bottomleft[0] - 0.1, bottomleft[1] - 0.06).property('dashed').draw()
    connect(conv1, 'east', conv_nodes[1], 'west')
    
    bottomleft = nodes[4].anchor('south west')
    width = nodes[5].anchor('east')[0] - bottomleft[0]
    height = nodes[3].anchor('north')[1] - bottomleft[1]
    conv3 = Rect(context, width + 0.2, height + 0.12).at(bottomleft[0] - 0.1, bottomleft[1] - 0.06).property('dashed').draw()
    connect(conv3, 'east', conv_nodes[3], 'west')

    x = x + 2.5
    y = 0
    node_width = 0.5
    node_height = 0.45
    hpad = 0.9
    vpad = (max_y - 4 * node_height - vec_height) / 4
    max_x = 0
    def draw_node(i, j, text):
        global max_x, max_y
        x_step = (node_width + hpad) / 2
        y_step = node_height + vpad
        right = x + j * x_step + node_width
        if right > max_x:
            max_x = right
        return Rect(context, node_width, node_height, text).at(x + j * x_step, y + i * y_step).draw()
    ssm_nodes = {
        1: draw_node(3, 2, '1'),
        2: draw_node(2, 1, '2'),
        3: draw_node(1, 1, '3'),
        4: draw_node(0, 0, '4'),
        5: draw_node(0, 2, '5'),
        6: draw_node(2, 3, '6'),
    }
    result2_x = ssm_nodes[1].anchor('north')[0] - 1
    result2_y = conv_nodes[1].anchor('south')[1]
    result2 = Rect(context, 2, vec_height, 'Tree Repr').at(result2_x, result2_y).draw()
    blend1_center = (ssm_nodes[2].anchor('east') + ssm_nodes[6].anchor('west')) / 2
    blend1 = Blend(context).at(*blend1_center).scale(0.3).draw()
    connect(ssm_nodes[2], 'east', blend1, 'west')
    connect(ssm_nodes[6], 'west', blend1, 'east')
    connect(blend1, 'north', ssm_nodes[1], 'south')
    blend2_center = (ssm_nodes[4].anchor('east') + ssm_nodes[5].anchor('west')) / 2
    blend2 = Blend(context).at(*blend2_center).scale(0.3).draw()
    connect(ssm_nodes[4], 'east', blend2, 'west')
    connect(ssm_nodes[5], 'west', blend2, 'east')
    connect(blend2, 'north', ssm_nodes[3], 'south')
    connect(ssm_nodes[3], 'north', ssm_nodes[2], 'south')
    connect(ssm_nodes[1], 'north', result2, 'south')
    print(max_x)

    context.save('figures/node_mixer.pdf')