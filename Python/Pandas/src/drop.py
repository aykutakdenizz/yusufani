import pandas as pd
films = pd.read_csv("imdb-1000.csv")
print(films.columns)
films= films.drop("content_rating",axis=1)
print(films.columns)

rowsToDrop = [0,1,3,4,5,6,8,9]
films = films.drop(rowsToDrop,axis=0)
print(films)