# -*- coding: utf-8 -*-

matris = [[1,3,45],[23,2,1],[2,4,1]]
matris2 = [[2,4,4],[13,2,1],[2,4,1]]
sonuc = [[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(matris)):
    for j in range(len(matris2)):
        sonuc[i][j]=matris[i][j]+matris2[i][j]
        
