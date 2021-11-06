import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# Модель второго уровня
from lightgbm import LGBMClassifier


class Recommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, user_features, item_features, weighting=True, n_lvl_2_train_weeks=4, n_lvl_2_candidates=50):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.data = data[data['item_id'] != 999999]
        self.user_features = user_features
        self.item_features = item_features

        self.user_item_matrix = self._prepare_matrix(self.data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.als_recommender = self.fit_als_recommender(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        self.fit_lvl_2_recommender(n_train_weeks=n_lvl_2_train_weeks, n_candidates=n_lvl_2_candidates)
        
    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
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

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit_als_recommender(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

        
    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]] if 999999 in self.itemid_to_id else None,
                                        recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.als_recommender, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.als_recommender.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]   # удалим юзера из запроса

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
 # Новый код для курсового ===>
    @staticmethod
    def split_by_weeks(data, n_last_weeks):
        split_week = data['week_no'].max() - n_last_weeks
        return (data[data['week_no'] <= split_week], data[data['week_no'] > split_week])

    @staticmethod
    def get_candidates(data, fn):
        r = pd.DataFrame(data['user_id'].unique(), columns = ['user_id']) 
        r['candidates'] = r['user_id'].apply(fn)
        s = r.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'
        r = r.drop('candidates', axis=1).join(s)
        return r

    
    def add_features(self, ds):

        ds = ds.merge(self.item_features, on='item_id', how='left')
        ds = ds.merge(self.user_features, on='user_id', how='left')
        ds = ds.merge(self.data, on=['user_id', 'item_id'], how='left')

        ds = ds.groupby(['user_id', 'item_id']).first().reset_index()

        baskets_by_user = self.data.groupby('user_id')['basket_id'].nunique().reset_index().rename(
            columns={'basket_id': 'baskets_by_user'})
        sales_value_by_user = self.data.groupby('user_id')['sales_value'].sum().reset_index().rename(
            columns={'sales_value': 'sales_value_by_user'})

        ds = ds.merge(baskets_by_user, on='user_id', how='left').merge(sales_value_by_user, on='user_id', how='left')
        ds['user_mean_check'] = ds['sales_value_by_user'] / ds['baskets_by_user'] 

        train_with_features = self.data.merge(self.item_features, on='item_id', how='left')

        quantity_by_user_commodity_desc = train_with_features.groupby(['user_id', 'commodity_desc'])['quantity'].sum().reset_index().rename(columns={'quantity': 'quantity_by_user_commodity_desc'})
        ds = ds.merge(quantity_by_user_commodity_desc, on=['user_id', 'commodity_desc'],
                      how='left').fillna({'quantity_by_user_commodity_desc':0})

        weeks_by_user = self.data.groupby('user_id')['week_no'].nunique().reset_index().rename(columns={'week_no': 'weeks_by_user'})
        ds = ds.merge(weeks_by_user, on='user_id', how='left')

        ds['baskets_per_week_by_user'] = ds['baskets_by_user'] / ds['weeks_by_user'] 

        weeks_by_item = self.data.groupby('item_id')['week_no'].nunique().reset_index().rename(columns={'week_no': 'weeks_by_item'})
        quanity_by_item = self.data.groupby('item_id')['quantity'].sum().reset_index().rename(columns={'quantity': 'quantity_by_item'})
        ds = ds.merge(weeks_by_item, on='item_id', how='left').merge(quanity_by_item, on='item_id', how='left')

        ds['quanity_per_week_by_item'] = ds['quantity_by_item'] / ds['weeks_by_item'] 

        cat_features = [f for f, t in zip(ds.dtypes.index, ds.dtypes) if t == 'object']    
        for c in cat_features:
            ds[c] = ds[c].astype('category')
        return ds

    @staticmethod
    def filter_by_users(users, data):
        return data[data['user_id'].isin(users['user_id'].unique())]
    
    def fit_lvl_2_recommender(self, n_train_weeks, n_candidates):
        """ Модель выбирает n_candidates товаров, купленных каждым пользователем,
            и отдает предпочтения товарам, похожим по lvl_2_model на купленные в течение последних n_train_weeks недель 
        """
        print('Подготовка 2-х уровневой модели ...')
        
        old, new = self.split_by_weeks(self.data, n_train_weeks)
 
        print(f'Выделено {len(new)} из {len(self.data)} записей для {n_train_weeks} последних недель.')
        print(f'Подготовка списка всех пользователей с кандидатами ({n_candidates} кандидатов для каждого пользователя) ...')
        all_users_with_candidates = self.get_candidates(self.data, lambda x: self.get_recommendations(0, x, N=n_candidates))
        print(f"Подготовлено {len(all_users_with_candidates)} записей {len(all_users_with_candidates['user_id'].unique())} пользователей с кандидатами.")
    
        print('Добавление признаков ...')
        data_with_features = self.add_features(all_users_with_candidates)
        print(f'Добавление признаков завершено, число записей: {len(data_with_features)} ...')
    
        new_users_data = self.filter_by_users(new, data_with_features)
        print(f'Число записей для пользователей из {n_train_weeks} последних недель: {len(new_users_data)} ...')
    
        targets = new[['user_id', 'item_id']].copy()
        targets['target'] = 1  
        targets = new_users_data.merge(targets, on=['user_id', 'item_id'], how='left')
        targets['target'].fillna(0, inplace= True)        
        
        print(f'Размер тренировочного набора: {len(targets)}')
        print(targets['target'].value_counts())

        X_train = targets.drop('target', axis=1)
        y_train = targets['target']
        
        cat_features = [f for f, t in zip(targets.dtypes.index, targets.dtypes) if str(t) == 'category']
        print(f'Категориальные признаки: {cat_features}')
        
        print('Тренируем LGBMClassifier ...')
        lgb = LGBMClassifier(objective='binary', max_depth=-1, num_leaves=2**16)
        lgb.fit(X_train, y_train, categorical_feature=cat_features)
        
        print('Готовим рекомендации LGBMClassifier ...')
        data_with_features['lvl_2_score'] = lgb.predict_proba(data_with_features)[:,1]
        lvl_2_scores = data_with_features.groupby(['user_id', 'item_id'])['lvl_2_score'].mean().reset_index()
        self.lvl_2_recommendations = lvl_2_scores.groupby('user_id').apply(
            lambda x: x.sort_values('lvl_2_score', ascending=False)['item_id'].tolist())
        print('Подготовка 2-х уровневой модели завершена')

    def get_lvl_2_recommendations(self, user, N=5):
        self._update_dict(user_id=user)
        r = self.lvl_2_recommendations[user][:N]
        if len(r) < N:
#            r += self.top_purchases[self.top_purchases['user_id'] == user]['item_id'].tolist()[:N - len(r)]
            r += self.get_own_recommendations(user, N)
        assert len(r) == N, 'Количество рекомендаций != {}'.format(N)
        return r

    def get_recommendations(self, level, user, N=5):
        """ Безопасный метод, работает для отсутствующих пользователей """
        r = []
        try:
            if level == -1 or level == 'top':
                r = self._extend_with_top_popular([], N)
            elif level == 0 or level == 'own':
                r = self.get_own_recommendations(user, N)
            elif level == 2 or level == 'lvl_2':
                r = self.get_lvl_2_recommendations(user, N)
            else:
                r = self.get_als_recommendations(user, N)
        except Exception as e:
#            print("exception", e)
            print(f'Нет рекомендации для user_id={user} для level={level}, возвращаем {N} самых популярных')
            r = self._extend_with_top_popular(r, N=N)
        assert len(r) == N, 'Количество рекомендаций != {}'.format(N)
        return r
 
    
    