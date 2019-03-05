# -*- coding: utf-8 -*-

number = int (input ("Please Enter the number"))
fakt=1
for i in range (number,1,-1):
    fakt*=i;
print(fakt)