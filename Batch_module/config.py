# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:43:06 2020

@author: sega01
"""

import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'