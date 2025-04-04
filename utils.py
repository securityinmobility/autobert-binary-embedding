#!/usr/bin/env python
# -*- coding: utf-8 -*-


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def find_first(haystack, needles):
    for needle in needles:
        if needle in haystack:
            return needle
    return None
