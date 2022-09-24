import gradio as gr
def function(movie):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.metrics.pairwise import cosine_similarity

    from sklearn.feature_extraction.text import CountVectorizer
    

    df = pd.read_csv("movie_metadata.csv")

    df.head()

    df['index'] = df.index
    df.head()

    def get_title_from_index(index):
        return df[df.index==index]["movie_title"].values[0]

    def get_index_from_title(title):
        return df[df.movie_title==title]["index"].values[0]

    sns.heatmap(df.isnull())

    df = df.fillna(" ")

    sns.heatmap(df.isnull())

    df.head()

    features = ["director_name","genres","actor_1_name","plot_keywords","country","language"]

    def combined_feautures(row):
        return row['director_name']+" "+row["genres"]+" "+row["actor_1_name"]+" "+row["plot_keywords"]+" "+row["country"]+" "+row["language"]+" "+str(row["imdb_score"])

    df["combined_feautures"] = df.apply(combined_feautures,axis=1)

    df.head()

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_feautures"])

    cosine_sim = cosine_similarity(count_matrix)

    movie_user_likes = str(movie)

    
    movie_index = get_index_from_title(movie_user_likes)

    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

    i=0
    str1 = ""
    for movie in sorted_similar_movies:
        str1 = str1+get_title_from_index(movie[0])+","
        i=i+1
        if i>20:
            break
    return str(str1)
def run(movie):
    movie = movie+"\xa0"
    final = function(movie)
    return final
    


outputs = gr.outputs.Textbox()

app = gr.Interface(fn=run, inputs="text", outputs=outputs,description="This is a movie recommendation model")
app.launch()
