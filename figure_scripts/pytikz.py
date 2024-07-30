#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta
import os

known_options = {
    "line cap": ["round", "rect", "butt"],
    "line join": ["round", "bevel", "miter"],
    ">": ["latex", "stealth"]
}

exclusive_groups = [
    ["ultra thin", "very thin", "thin", "semi",
        "thick", "very thick", "ultra thick"],
    ["solid", "dashed", "dotted", "dash dotted",
        "densely dotted", "loosely dotted", "double"],
    ["<-", "->", "<<-", "->>", "<->", "<->>", "<<->", "<<->>"],
    ["red", "green", "blue", "yellow", "black", "white", "gray", "brown"],
]


def found_group(option: str):
    for idx, group in enumerate(exclusive_groups):
        if option in group:
            return idx
    return -1


def set_properties(properties: dict[str, str], *args, **kwargs):
    for arg in args:
        properties[arg] = ''
    for key, value in kwargs.items():
        properties[key.replace('_', ' ')] = value


def merge_properties(properties: dict[str, str], *args, **kwargs):
    standalone_options = [
        key for (key, value) in properties.items() if value == '']
    groups_map = {}
    for idx, group in enumerate(exclusive_groups):
        for option in standalone_options:
            if option in group:
                groups_map[idx] = option
    for arg in args:
        group_idx = found_group(arg)
        if group_idx != -1 and group_idx in groups_map:
            del properties[groups_map[group_idx]]
        properties[arg] = ''
    for key, value in kwargs.items():
        properties[key.replace('_', ' ')] = value


def merge_other(properties: dict[str, str], other: dict[str, str]):
    standalone_options = [
        key for (key, value) in properties.items() if value == '']
    other_standalone_options = [
        key for (key, value) in other.items() if value == '']
    groups_map = {}
    for idx, group in enumerate(exclusive_groups):
        for option in standalone_options:
            if option in group:
                groups_map[idx] = option
    for arg in other_standalone_options:
        group_idx = found_group(arg)
        if group_idx != -1 and group_idx in groups_map:
            del properties[groups_map[group_idx]]
        properties[arg] = ''
    for key, value in other.items():
        if value != '':
            properties[key] = value


class draw_call_item(metaclass=ABCMeta):
    pass


class string_item(draw_call_item):
    def __init__(self, command: str) -> None:
        self.command = command


class coord_item(draw_call_item):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class draw_call:
    def __init__(self, command: str) -> None:
        self.command = command
        self.items: list[draw_call_item] = []
        self.properties: dict[str, str] = {}

    def property(self, *args, **kwargs) -> 'draw_call':
        merge_properties(self.properties, *args, **kwargs)
        return self

    def _set_property(self, properties: dict[str, str]) -> None:
        self.properties = properties.copy()

    def _merge(self, properties: dict[str, str]) -> 'draw_call':
        call = draw_call(self.command)
        call.items = self.items.copy()
        call.properties = self.properties.copy()
        merge_other(call.properties, properties)
        return call


class tikz_context:
    def __init__(self) -> None:
        self.draw_calls: list[tuple[draw_call, float, float, float]] = []
        self.libraries: list[str] = []
        self.colors: list[tuple[str, tuple[int, int, int]]] = []

    @staticmethod
    def _create_draw_call(command: str, *args) -> draw_call:
        call = draw_call(command)
        for arg in args:
            if isinstance(arg, tuple):
                call.items.append(coord_item(arg[0], arg[1]))
            elif isinstance(arg, str):
                call.items.append(string_item(arg))
            elif isinstance(arg, coord):
                call.items.append(coord_item(arg.x, arg.y))
            else:
                raise TypeError(f'Invalid argument type: {type(arg)}')
        return call

    @staticmethod
    def _replace_template(template: str, scale: float) -> str:
        idx = 0
        ret = ''
        while idx < len(template):
            if template[idx] == '{':
                if idx + 1 < len(template) and template[idx + 1] == '{':
                    ret += '{'
                    idx += 2
                else:
                    idx_end = idx + 1
                    while idx_end < len(template) and template[idx_end] != '}':
                        idx_end += 1
                    if idx_end == len(template):
                        raise ValueError('Invalid template')
                    ret += str(float(template[idx + 1:idx_end]) * scale)
                    idx = idx_end + 1
            elif template[idx] == '}':
                if idx + 1 < len(template) and template[idx + 1] == '}':
                    ret += '}'
                    idx += 2
                else:
                    raise ValueError('Invalid template')
            else:
                ret += template[idx]
                idx += 1
        return ret
    
    def _push_draw_call(self, draw_call: draw_call, x: float, y: float, scale: float) -> None:
        self.draw_calls.append((draw_call, x, y, scale))

    def _draw_impl(self, draw_call: draw_call, x: float, y: float, scale: float) -> str:
        draw_call_str = draw_call.command
        if draw_call.properties:
            draw_call_str += ' ['
            for (key, value) in draw_call.properties.items():
                if value == '':
                    draw_call_str += f'{key},'
                else:
                    draw_call_str += f'{key}={value},'
            draw_call_str = draw_call_str[:-1] + ']'
        for item in draw_call.items:
            if isinstance(item, string_item):
                draw_call_str += f' {tikz_context._replace_template(item.command, scale)}'
            elif isinstance(item, coord_item):
                draw_call_str += f' ({x + item.x * scale}, {y + item.y * scale})'
        draw_call_str += ';'
        return draw_call_str

    def draw(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\draw', *args)
        self._push_draw_call(call, 0, 0, 1)
        return self.draw_calls[-1][0]

    def fill(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\fill', *args)
        self._push_draw_call(call, 0, 0, 1)
        return self.draw_calls[-1][0]
    
    def filldraw(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\filldraw', *args)
        self._push_draw_call(call, 0, 0, 1)
        return self.draw_calls[-1][0]

    def node(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\node', *args)
        self._push_draw_call(call, 0, 0, 1)
        return self.draw_calls[-1][0]

    def coordinate(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\coordinate', *args)
        self._push_draw_call(call, 0, 0, 1)
        return self.draw_calls[-1][0]
    
    def use(self, lib: str) -> None:
        self.libraries.append(lib)

    def add_color(self, name: str, color: tuple[int, int, int]|str) -> None:
        if isinstance(color, str):
            assert color[0] == '#' and len(color) == 7, 'Invalid color'
            color = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        self.colors.append((name, color))

    def generate(self) -> str:
        begin_str = '\\documentclass[tikz]{standalone}'
        document_begin_str = \
'''\\begin{document}
\\begin{tikzpicture}'''
        end_str = \
'''\\end{tikzpicture}
\\end{document}'''
        lib_str = '\n'.join([f'\\usetikzlibrary{{{lib}}}' for lib in self.libraries])
        color_str = '\n'.join([f'\\definecolor{{{name}}}{{RGB}}{{{R},{G},{B}}}' for (name, (R, G, B)) in self.colors])
        draw_call_strs = '\n'.join([self._draw_impl(call, x, y, scale) for (call, x, y, scale) in self.draw_calls])

        return '\n'.join([begin_str, lib_str, color_str, document_begin_str, draw_call_strs, end_str])
    
    def save(self, path: str) -> None:
        dest_dir = os.path.dirname(path)
        if dest_dir != '' and not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        if dest_dir == '':
            dest_dir = '.'
        tex_path = os.path.splitext(path)[0] + '.tex'
        log_path = os.path.join(dest_dir, 'latexmk.log')
        with open(tex_path, 'w') as f:
            f.write(self.generate())
        os.system(f'latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir={dest_dir} -shell-escape {tex_path} > {log_path} 2>&1')

class coord:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    
    def __add__(self, other: 'coord') -> 'coord':
        if isinstance(other, tuple):
            print(other)
            return coord(self.x + other[0], self.y + other[1])
        return coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'coord') -> 'coord':
        if isinstance(other, tuple):
            return coord(self.x - other[0], self.y - other[1])
        return coord(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other: float) -> 'coord':
        return coord(self.x * other, self.y * other)
    
    def __rmul__(self, other: float) -> 'coord':
        return coord(self.x * other, self.y * other)
    
    def __truediv__(self, other: float) -> 'coord':
        return coord(self.x / other, self.y / other)
    
    def __sizeof__(self) -> int:
        return 2
    
    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError('Invalid index')

class shape_instance:
    def __init__(self, shape: 'tikz_shape') -> None:
        self.shape = shape
        self.scale_ = 1
        self.x = 0
        self.y = 0
        self.properties = {}

    def at(self, x, y) -> 'shape_instance':
        self.x = x
        self.y = y
        return self

    def property(self, *args, **kwargs) -> 'shape_instance':
        merge_properties(self.properties, *args, **kwargs)
        return self

    def scale(self, scale: float) -> 'shape_instance':
        self.scale_ = scale
        return self

    def anchor(self, path: str) -> coord:
        item_list = path.split('.')
        attrs = item_list[:-1]
        anchor_name = item_list[-1]
        instances = [self]
        for attr in attrs:
            ins = getattr(instances[-1].shape, attr)
            assert isinstance(ins, shape_instance)
            instances.append(ins)
        x, y = instances[-1].shape._anchors()[anchor_name]
        for ins in instances[::-1]:
            x = ins.x + x * ins.scale_
            y = ins.y + y * ins.scale_
        return coord(x, y)

    def custom_anchor(self, *args, path = None, **kwargs) -> coord:
        instances = [self]
        if path != None:
            attrs = path.split('.')
            for attr in attrs:
                ins = getattr(instances[-1].shape, attr)
                assert isinstance(ins, shape_instance)
                instances.append(ins)
        x, y = instances[-1].shape._custom_anchor(*args, **kwargs)
        for ins in instances[::-1]:
            x = ins.x + x * ins.scale_
            y = ins.y + y * ins.scale_
        return coord(x, y)

    def draw(self) -> 'shape_instance':
        for draw_call, x, y, scale in self.shape.draw_calls:
            new_x = self.x + x * self.scale_
            new_y = self.y + y * self.scale_
            new_scale = self.scale_ * scale
            self.shape.context._push_draw_call(
                draw_call._merge(self.properties), new_x, new_y, new_scale)
        return self


class tikz_shape(tikz_context):
    def __init__(self, context: tikz_context) -> None:
        self.context = context
        self.draw_calls: list[tuple[draw_call, float, float, float]] = []
        self.current_properties: dict[str, str] = {}

    def _push_draw_call(self, _draw_call: draw_call, x: float, y: float, scale: float) -> None:
        self.draw_calls.append((_draw_call, x, y, scale))

    def _anchors(self) -> dict[str, tuple[float, float]]:
        return {}

    def _custom_anchor(self, *args, **kwargs) -> tuple[float, float]:
        raise NotImplementedError("_custom_anchor")

    def set_property(self, *args, **kwargs) -> None:
        self.current_properties = {}
        set_properties(self.current_properties, *args, **kwargs)

    def clear_property(self) -> None:
        self.set_property()

    def draw(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\draw', *args)
        call._set_property(self.current_properties)
        self.draw_calls.append((call, 0, 0, 1))
        return self.draw_calls[-1][0]

    def fill(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\fill', *args)
        call._set_property(self.current_properties)
        self.draw_calls.append((call, 0, 0, 1))
        return self.draw_calls[-1][0]

    def filldraw(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\filldraw', *args)
        call._set_property(self.current_properties)
        self.draw_calls.append((call, 0, 0, 1))
        return self.draw_calls[-1][0]
    
    def node(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\node', *args)
        call._set_property(self.current_properties)
        self.draw_calls.append((call, 0, 0, 1))
        return self.draw_calls[-1][0]
    
    def coordinate(self, *args) -> draw_call:
        call = tikz_context._create_draw_call(r'\coordinate', *args)
        call._set_property(self.current_properties)
        self.draw_calls.append((call, 0, 0, 1))
        return self.draw_calls[-1][0]

    def at(self, x, y) -> shape_instance:
        return shape_instance(self).at(x, y)
    
    def scale(self, scale: float) -> shape_instance:
        return shape_instance(self).scale(scale)

    def property(self, *args, **kwargs) -> shape_instance:
        return shape_instance(self).property(*args, **kwargs)