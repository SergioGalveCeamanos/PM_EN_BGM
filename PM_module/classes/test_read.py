# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:08:56 2020

@author: sega01
"""

f = open("msos.txt", "r")
msos=[]
for x in f:
    a=x.split()
    l=[]
    for new in a[2:]:
        l.append(int(new))
    msos.append(l)
print(msos)