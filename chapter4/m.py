import os
import pandas as pd
data_folder = os.path.join(os.path.expanduser('./'), 'Data', 'ml-100k')
ratings_filename = os.path.join(data_folder, "u.data")

all_ratings = pd.read_csv(ratings_filename, delimiter="\t", header=None, names = ["UserID", "MovieID", "Rating", "Datetime"])

all_ratings['Favorable'] = all_ratings['Rating'] > 3

print(all_ratings[10:15])

ratings = all_ratings[all_ratings['UserID'].isin(range(200))]

favorable_ratings = ratings[ratings['Favorable']]

favorable_reviews_by_users = dict((k, frozenset(v.values))
                                    for k, v in favorable_ratings 
                                    groupby("UserID")["MovieID"])

num_favorable_by_movie = ratings[["MovieID", "Favorable"]]. groupby("MovieID").sum()

print(num_favorable_by_movie.sort("Favorable", ascending=False)[:5])
