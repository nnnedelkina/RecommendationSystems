import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, 
                 weighting=True, 
                 factors=20,
                 regularization=0.001,
                 iterations=15,
                 num_threads=4,
                 user_id_column='user_id', 
                 item_id_column='item_id', 
                 value_column='quantity'):
        
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.num_threads = num_threads

        self.user_id_column=user_id_column
        self.item_id_column=item_id_column
        self.value_column = value_column
        
        
        self.data = data
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
 



    def prepare_matrix(self, data):

        user_item_matrix = pd.pivot_table(data, 
                                  index=self.user_id_column, columns=self.item_id_column, 
                                  values=self.value_column, # Можно пробовать другие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
    
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    def fit_own_recommender(self, user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=self.num_threads)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    def fit(self, user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=self.factors, 
                                             regularization=self.regularization,
                                             iterations=self.iterations,  
                                             num_threads=self.num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model

    def get_popular_items_for_user(self, user_id, N=5):
        user_items = self.data[self.data[self.user_id_column] == user_id]
        top_user_items = user_items.groupby([self.item_id_column])[self.value_column].count().reset_index()
        top_user_items = top_user_items[top_user_items[self.item_id_column] != 999999]
        top_user_items.sort_values(self.value_column, ascending=False, inplace=True)
        topn_user_items=top_user_items.head(N)
        return topn_user_items[self.item_id_column].tolist()

    def get_similar_items(self, item_id, N=5):
        similar_items = self.model.similar_items(self.itemid_to_id[item_id], N=N+1)
        similar_items = [self.id_to_itemid[sim[0]] for sim in similar_items[1:1+N]]
        return similar_items
    
    def get_recommendations(self, user_id, N=5, model=None):

        if model == None:
            model = self.model
            
        res = [self.id_to_itemid[rec[0]] for rec in 
                model.recommend(userid=self.userid_to_id[user_id], 
                                user_items=csr_matrix(self.user_item_matrix).tocsr(),   # на вход user-item matrix
                                N=N, 
                                filter_already_liked_items=False, 
                                filter_items=[self.itemid_to_id[999999]] if 999999 in self.itemid_to_id else None,  # !!! 
                                recalculate_user=True)]
        return res
    
    def get_similar_items_recommendation(self, user_id, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        topn_user_items = self.get_popular_items_for_user(user_id)
        res = self.get_similar_items(topn_user_items[0], 1 + N - len(topn_user_items)) 
        for iid in topn_user_items[1:]:
            res += self.get_similar_items(iid, 1)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user_id, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        similar_users = [ self.id_to_userid[u[0]] for u in self.model.similar_users(self.userid_to_id[user_id], N=N+1) ][1:]
        res = self.get_recommendations(similar_users[0], 1 + N - len(similar_users), self.own_recommender) 
        for uid in similar_users[1:]:
            res += self.get_recommendations(uid, 1, self.own_recommender) 

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


