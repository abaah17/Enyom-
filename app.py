import streamlit as st
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
import pandas as pd
import pickle

Model = pickle.load(open('algo.pkl', 'rb'))
music = pd.read_pickle('music.csv')
music_pivot = pd.read_pickle('music_pivot.csv')


# knn=NearestNeighbors(n_neighbors=10,metric='cosine')
# Model=knn.fit(music_pivot)

def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:#fcc930;padding:20px;font-weight:15px"> 
    <h1 style ="color:white;text-align:center;"> Enyom Recommendation System</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_value_goes_here = ""
    user_ids = st.number_input("Enter the user id you want to recommend for", 0, 100000000, 0)
    num_songs = st.number_input("How many songs do you want to predict", 0, 100000000, 0)

    num_of_ids = music_pivot.shape[0]
    # print(num_of_ids)
    result = ""

    # Display Books
    if st.button("Predict"):
        # global result


        if 0 <= int(user_ids) <= num_of_ids:
            print(user_ids)
            user = music_pivot.iloc[int(user_ids)]
            print(user)

            user = user.astype(float)

            distances, indices = Model.kneighbors([user])

            print(distances, indices)

            neighbors = []
            for item in indices[0][1:]:
                neighbors.append(music[music.index == item].user_id.values[0])

            # Make a dataframe with details of only the neighbors.
            neighbor_songs = pd.DataFrame(columns=['user_id', 'song_id', 'listen_count', 'title', 'artist', 'song'])
            for item in neighbors:
                neighbor_songs = neighbor_songs.append(music[music.user_id == item], ignore_index=True)

            neighbor_songs = pd.DataFrame(
                {'Count': neighbor_songs['listen_count'], 'Song': neighbor_songs.song.tolist()})

            neighbor_songs = neighbor_songs.sort_values('Count', ascending=False)

            total_songs = len(neighbor_songs)

            if 0 <= int(num_songs) <= total_songs:
                result = neighbor_songs['Song'][0:int(num_songs)]
            else:
                result = "Choose a number that is less than " + str(total_songs + 1)

    else:
        result = "Choose an ID that is between " + str(0) + "  and  " + str(num_of_ids)

    st.write(result)


if __name__ == '__main__':
    main()
