#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psyclab/sl/cstruct.py
Vince Enachescu 2019
"""

import re
import ctypes

from functools import reduce
from operator import mul

import CppHeaderParser

# slightly disgusting hack to make up for lack of unsigned char type
ctypes.c_uchar = ctypes.c_char


def convert_bytes_to_structure(st, byte):
    # sizoef(st) == sizeof(byte)
    ctypes.memmove(ctypes.addressof(st), byte, ctypes.sizeof(st))


def convert_struct_to_bytes(st):
    buffer = ctypes.create_string_buffer(ctypes.sizeof(st))
    ctypes.memmove(buffer, ctypes.addressof(st), ctypes.sizeof(st))
    return buffer.raw


def convert_int_to_bytes(number, size):
    return (number).to_bytes(size, 'big')


def parse_defines(header, include_enums=False):

    defines, enums = {}, {}
    for enum in header.enums:
        values = {v['value']: v['name'] for v in list(enum['values'])}
        enums[enum['name']] = values
        defines.update({value: key for key, value in values.items()})

    for define in header.defines:
        if ' ' in define:
            # key, value = define.split(' ')
            # key, value, *_ = define.split()
            key, value = define.split()[:2]
        else:
            key, value = define, None
        defines[key] = value

    if not include_enums:
        return replace_defines(defines, defines)
    return replace_defines(defines, defines), enums


def replace_defines(value, defines):
    """ try to replace any preprocessor defines with their values """

    if isinstance(value, dict):
        return {key: replace_defines(item, defines) for key, item in value.items()}

    if not isinstance(value, str):
        return value

    match = re.search('\b*?([A-Z]\w{2,})\b?', value)
    if match:
        define = match.group(1)
        if define in defines.keys():
            value = value.replace(define, '{}'.format(defines[define]))

    try:
        return eval(value)
    except:
        return value


def read_struct_format(header_path, class_name, defines=None, types=None):

    defines, types = defines or {}, types or {'uint64_t': ctypes.c_ulonglong}

    if '\n' in header_path:
        header = header_path
    else:
        with open(header_path, 'r') as fp:
            header = fp.read()
    lines = header.split('\n')

    cpp_header = CppHeaderParser.CppHeader(header, argType='string')
    defines.update(parse_defines(cpp_header))
    properties = cpp_header.classes[class_name]['properties']

    fields = []
    for kind in ('public', 'protected', 'private'):
        for field in properties.get(kind, []):
            field['line'] = lines[field['line_number'] - 1]
            fields.append(field)

    def parse_field(field):

        ctype = field.get('ctypes_type')
        if ctype is not None:
            ctype = eval(ctype)
        else:
            ctype = types[field.get('raw_type')]

        # parse out the array sizes if present
        if field['array']:
            match = re.search('{}(\[.+?\])(\[.+?\])*'.format(field['name']), field['line'])
            array = [replace_defines(s.strip('[]'), defines) for s in match.groups() if s is not None]
            return (field['name'], ctype * reduce(mul, array))

        return field['name'], ctype

    return list(map(parse_field, sorted(fields, key=lambda item: item['line_number'])))


class CStruct(ctypes.Structure):

    @property
    def raw_bytes(self):
        return convert_struct_to_bytes(self)

    @property
    def members(self):
        return [field[0] for field in self._fields_]

    @property
    def values(self):
        return {key: getattr(self, key) for key in self.members}
