import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

#    # Уберем самые популярные товары (их и так купят)
#    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
#    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features.\
                                        groupby('department')['item_id'].nunique().\
                                        sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]


    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 0.1]

#    # Уберем слишком дорогие товары
#    data = data[data['price'] < 50]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    popularity.index = popularity.index.astype('int')
    top = popularity.sort_values('n_sold', ascending=False).head(int(take_n_popular)).item_id.tolist()
    
    # Заведем фиктивный item_id для товаров, менее популярных, чем take_n_popular
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    # ...

    data = data[data['item_id'] != 999999]
    
    return data


def postfilter_items(user_id, recommednations):
    pass