# -*- coding: utf-8 -*-

counter = int (input("How many line with starts you want to write ? " ))
stars= ""
for x in range(1,counter+1):
    stars = stars + "*"
    print(stars)