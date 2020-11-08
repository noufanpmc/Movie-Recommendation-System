import pandas
import numpy as np
import random

from .recommender_system_base import RecommenderSystemBase

class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        
    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))
    
    def __setitem__(self, inp_vec, label):
        #print("generating hash value for hash table generation")
        hash_value = self.generate_hash(inp_vec)
        #print("completed generating hash value for hash table generation")
        self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]
        #print("completed Setting values to hash table")
    def __getitem__(self, inp_vec):
        #print("generating hash value for searching hash table")
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
            
    def __setitem__(self, inp_vec, label):
        #print("generating hash tables")
        for table in self.hash_tables:
            table[inp_vec] = label
        #print("hash table generation complete")
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            #print("looping through hash tables")
            results.extend(table[inp_vec])
            #print("suggestions from table ",results)
          
        return list(set(results))


class ItemItemRecommenderSystem(RecommenderSystemBase):
    """
    Attributes
    ----------

    Methods
    -------
    compute_movie_embeddings
        Computes the movie embeddings.

    recommend_similar_movies
        Recommends the k most similar of the movie with the id 'movie_id'.

    recommend_movies_to_user
        Given a user with a watch history, it recommends the k movies that he will most likely watch.

    get_movies_embeddings
        Returns the embedding of the movies with movie_id in movie_ids.

    Notes
    -----
    - You can add other attributes and methods to this class.
    - In the constructor parameters, you can add other datasets if you need them.

    Examples
    --------
    >>> rec_sys = ItemItemRecommenderSystem(**kwargs)
    >>> ...
    >>> rec_sys.recommend_similar_movies(movie_id='the_promise-das_versprechen-en-1995', k=10)
    ...
    >>> rec_sys.recommend_movies_to_user(user_id=25, k=10)
    ...
    >>> movie_embeddings = rec_sys.get_movies_embeddings(movie_ids)
    >>> visualize_embeddings(movie_embeddings)
    ...
    """

    def __init__(self, ratings_dataframe: pandas.DataFrame, movies_metadata_dataframe: pandas.DataFrame,
                 keywords_dataframe: pandas.DataFrame, credits_dataframe: pandas.DataFrame) -> None:
        """Sets the movie_embeddings attribute.

        Parameters
        ----------
        ratings_dataframe : pandas.DataFrame
            The movie ratings of users.
        movies_metadata_dataframe : pandas.DataFrame
            The movies metadata.
        keywords_dataframe : pandas.DataFrame
            The movies keywords.
        credits_dataframe : pandas.DataFrame
            The movies credits.
        """
        print("starting init")
        super().__init__(ratings_dataframe, movies_metadata_dataframe, keywords_dataframe, credits_dataframe)
        self.movie_embeddings = self.make_embeddings(self.movies_dataframe, 'movie')
        self.user_embeddings = self.make_embeddings(self.movies_dataframe, 'user')
        print("ending init")

    def make_embeddings(self, merged_movie_data, emb_type):
        print("make embedding")

        if emb_type == 'movie':
            ratings_matrix = merged_movie_data.pivot_table(columns='userId',index='movie_id',values='rating')
        else:
            ratings_matrix = merged_movie_data.pivot_table(columns='movie_id',index='userId',values='rating')

        ratings_matrix.dropna(axis=1, how='all', inplace=True)
        ratings_matrix.fillna( 0, inplace = True )
        print("ending embedding")
        return ratings_matrix

    def getJaccardSim(self, movie1,movie2):
        movie1=''.join((movie1 > 3).astype(int).astype('str'))
        movie2=''.join((movie2 > 3).astype(int).astype('str'))
        N = 0
        D = 0
        for i in range(len(movie1)):
            sum_ = int(movie1[i]) + int(movie2[i])
            if(sum_ >= 1):
                flag = 1
                D += 1
                if(sum_ == 2):
                    N += 1
        if D == 0:
            return 0
        return(float(N)/D)
        
    def getCosineSim(self, movie1,movie2):
        return np.dot(movie1,movie2)/(np.linalg.norm(movie1)*np.linalg.norm(movie2))


    def recommend_movies_to_user(self, user_id: int, k: int, algo) -> pandas.DataFrame:
        """Given a user with a watch history, it recommends the k movies that he will most likely watch.

        user_favourite_movies = the set of movies that the user watched and liked.

        If len(user_favourite_movies) = 0:
            Recommend k random movies from the set of highly rated movies in the dataset.
            These k movies should be chosen randomly. So if the function is executed 2 times, it should
            return different results.

        If k < len(user_favourite_movies):
            Select a random set of movies from the user_favourite_movies set and recommend a movie for each item.

        If k > len(user_favourite_movies):
            Select n movies for each movie the user liked.
            Example :
                k = 10 and len(user_favourite_movies) = 1
                    Recommend 10 movies that are similar to the movie the user watched.
                k = 10 and len(user_favourite_movies) = 3
                    Recommend:
                        3 movies that are similar the 1st movie the user liked.
                        3 movies that are similar the 2nd movie the user liked.
                        4 movies that are similar the 3rd movie the user liked.

        Parameters
        ----------
        user_id : int
            The id of the user
        k : int
            The number of movies to recommend

        Returns
        -------
        pandas.DataFrame
            A subset of the movies_dataframe with the k movies that the user may like.
        """

        from scipy.sparse import csr_matrix
        embeddings_sparse = csr_matrix(self.movie_embeddings.values)

        from sklearn.neighbors import NearestNeighbors

        user_favourite_movies = self.movies_dataframe[self.movies_dataframe.userId == user_id][self.movies_dataframe.rating >= 3].movie_id.tolist()
        #print("favorite movies",user_favourite_movies)      

        if len(user_favourite_movies) == 0:
            return self.movies_dataframe[self.movies_dataframe.rating >= 4].sample(k)

        elif algo == 'KNN':

            if k < len(user_favourite_movies):
                user_favourite_movies = random.sample(user_favourite_movies, k)
                model = NearestNeighbors(n_neighbors=k,algorithm='brute',metric='cosine')
                model.fit(embeddings_sparse)        

                movie_embeddings = self.get_movies_embeddings(user_favourite_movies) 
                distances,suggestions=model.kneighbors(movie_embeddings.values)
                
                movies = []
                distance = []
                for i in user_favourite_movies:
                    movie_embeddings = self.get_movies_embeddings(i) 
                    distances,suggestions=model.kneighbors(movie_embeddings.values.reshape(1, -1),2)
                    distances= distances.flatten()
                    suggestions= suggestions.flatten()
                    
                    for i in range(1,len(suggestions)):
                        movie_id=self.movie_embeddings.index[suggestions[i]]
                        movies.append(movie_id)
                        distance.append(distances[i])   

                return self.movies_dataframe.loc[self.movies_dataframe['movie_id'].isin(movies)].drop_duplicates(subset=['movie_id'])
                    

            elif k > len(user_favourite_movies):
                n = len(user_favourite_movies)
                q = k//n
                r = k%n
                k_values = []
                for _ in range(n):
                    k_values.append(q)
                k_values[-1] += r      

                movies = []
                distance = []
                
                model = NearestNeighbors(n_neighbors=k_values[-1],algorithm='brute',metric='cosine')
            
                model.fit(embeddings_sparse)
                

                for idx,i in enumerate(k_values):
                    movie_embeddings = self.get_movies_embeddings(user_favourite_movies[idx]) 
                    distances,suggestions=model.kneighbors(movie_embeddings.values.reshape(1, -1),i+1)
                    distances= distances.flatten()
                    suggestions= suggestions.flatten()
                    
                    for i in range(1,len(suggestions)):
                        movie_id=self.movie_embeddings.index[suggestions[i]]
                        movies.append(movie_id)
                        distance.append(distances[i])   
                
                return self.movies_dataframe.loc[self.movies_dataframe['movie_id'].isin(movies)].drop_duplicates(subset=['movie_id'])

        else:
            if k < len(user_favourite_movies):
                user_favourite_movies = random.sample(user_favourite_movies, k)
                movies = []
                for user_fav in user_favourite_movies:
                    res = self.recommend_similar_movies(user_fav, 1, algo)
                    movies.append(res.movie_id.values[0])
                return self.movies_dataframe.loc[self.movies_dataframe['movie_id'].isin(movies)].drop_duplicates(subset=['movie_id'])
            
            elif k > len(user_favourite_movies):
                n = len(user_favourite_movies)
                q = k//n
                r = k%n
                k_values = []
                for _ in range(n):
                    k_values.append(q)
                k_values[-1] += r

                movies = []
                for i in range(n):
                    res = self.recommend_similar_movies(user_favourite_movies[i], k_values[i], algo)
                    movies.extend(res.movie_id.tolist())

                return self.movies_dataframe.loc[self.movies_dataframe['movie_id'].isin(movies)].drop_duplicates(subset=['movie_id'])
                


    def recommend_similar_movies(self, movie_id: str, k: int, algo) -> pandas.DataFrame:
        """Recommends the k most similar movies of the movie with the id 'movie_id'.

        Parameters
        ----------
        movie_id : str
            The id of the movie.
        k : int
            The number of similar movies to recommend.

        Returns
        -------
        pandas.DataFrame
            A subset of the movies_dataframe with the k similar movies of the target movie (movie_id).
        """

        if algo == 'knn':

            from scipy.sparse import csr_matrix
            embeddings_sparse = csr_matrix(self.movie_embeddings)

            from sklearn.neighbors import NearestNeighbors
            model = NearestNeighbors(n_neighbors=k,algorithm='brute',metric='cosine')

            model.fit(embeddings_sparse)
        
            #condition = self.movies_dataframe['movie_id']==movie_id
            #idVal= self.movies_dataframe[condition].drop_duplicates(subset=['movie_id'])['movieId']
            #print("Movie id", idVal)
            movie_embeddings = self.get_movies_embeddings(movie_id)
            distances,suggestions=model.kneighbors(movie_embeddings.values.reshape(1,-1),k+1)
            suggestions= suggestions.flatten()
    
            print(suggestions)
            movies = []
            for i in range(1,len(suggestions)):
                movies.append(self.movie_embeddings.index[suggestions[i]])

            return self.movies_dataframe.loc[self.movies_dataframe['movie_id'].isin(movies)].drop_duplicates(subset=['movie_id'])

        else:
            nusers = self.movie_embeddings.columns
            nmovies = self.movie_embeddings.index

            hash_table = LSH(num_tables=20,hash_size=10, inp_dimensions=len(nusers))

            for i in range(len(nmovies)):
                hash_table[self.movie_embeddings.loc[nmovies[i]]]=nmovies[i]

            inp_vec=self.movie_embeddings.loc[movie_id]
            # print("Movie_id" ,nmovies[movie_id])
            similar_movies = hash_table[inp_vec]
            cos_sim_values =[]
            jac_sim_values=[]

            for a in similar_movies:
                if a== movie_id:
                    continue
                out_vec = self.movie_embeddings.loc[a]
                cos_sim_values.append(self.getCosineSim(inp_vec,out_vec))
                jac_sim_values.append(self.getJaccardSim(inp_vec,out_vec))
            
            if algo == 'LSH-C':
                ranked_cos_sim = np.argsort(np.array(cos_sim_values))
                movies_id_cos = ranked_cos_sim[::-1][:k]
                cos_sugg = []
                for i in range(0,k):
                    movie_sugg_cos = similar_movies[movies_id_cos[i]]
                    cos_sugg.append(self.movies_dataframe[self.movies_dataframe["movie_id"]==str(movie_sugg_cos)]["movie_id"].values[0])
                return self.movies_dataframe.loc[self.movies_dataframe["movie_id"].isin(cos_sugg)].drop_duplicates(subset=['movie_id'])

            elif algo == 'LSH-J':
                ranked_jac_sim = np.argsort(np.array(jac_sim_values))
                movies_id_jac = ranked_jac_sim[::-1][:k]
                jac_sugg = []
                for i in range(0,k):
                    movie_sugg_jac= similar_movies[movies_id_jac[i]]
                    jac_sugg.append(self.movies_dataframe[self.movies_dataframe["movie_id"]==str(movie_sugg_jac)]["movie_id"].values[0])
                return self.movies_dataframe.loc[self.movies_dataframe["movie_id"].isin(jac_sugg)].drop_duplicates(subset=['movie_id'])


    def get_movies_embeddings(self, movie_ids: [str]) -> pandas.DataFrame:
        """Returns the embedding of the movies with movie_id in movie_ids.

        Parameters
        ----------
        movie_ids : [str]
            List of the movies movie_id.

        Returns
        -------
        pandas.DataFrame
            The embeddings of the movies with movie_id in movie_ids.
        """
       
        return self.movie_embeddings.loc[movie_ids,:]