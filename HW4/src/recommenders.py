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

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        # ИЗВИНИТЕ, ПОКА ЗАГЛУШКА, ЧЕРЕЗ ЧАС БУДЕТ

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        # your_code
        # ИЗВИНИТЕ, ПОКА ЗАГЛУШКА, ЧЕРЕЗ ЧАС БУДЕТ

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


