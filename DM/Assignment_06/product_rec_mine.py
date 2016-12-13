import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

# 去除了'tipodom'特征因为取值都是1
# 去除了'ult_fec_cli_1t' 'fecha_alta'
# 'fecha_dato'与'ncodpers'单独处理
features_list = ['ind_empleado', 'pais_residencia', 'sexo', 'age',
                 'ind_nuevo', 'antiguedad', 'indrel','indrel_1mes',
                 'tiprel_1mes', 'indresi', 'indext', 'conyuemp',
                 'canal_entrada', 'indfall', 'cod_prov', 'nomprov',
                 'ind_actividad_cliente', 'renta', 'segmento']

products_list = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

continuous_features = ['age', 'antiguedad', 'renta']
discrete_features = list(set(features_list) ^ set(continuous_features))


def handle_nan(df_col, v):
    return df_col.map(lambda x: v if pd.isnull(x) or str(x).strip() == 'NA' or str(x).strip() == '' else x)


def handle_age(df):
    print("开始处理 age...")
    # 格式问题，处理空格与字符串年龄
    df["age"] = handle_nan(df["age"], np.NaN)
    df["age"] = df["age"].map(lambda x: int(x.strip()) if (isinstance(x, str)) else x)
    # 18-100岁区间之外的数据预处理; 缺失值填充
    df.loc[df.age < 18, "age"] = df.loc[(df.age >= 18) & (df.age <= 20), "age"].mean(skipna=True)
    df.loc[df.age > 100, "age"] = df.loc[(df.age >= 98) & (df.age <= 100), "age"].mean(skipna=True)
    df["age"].fillna(df["age"].mean(), inplace=True)
    # 规格化至[0, 1]
    age_min = df["age"].min()  # 18
    age_max = df["age"].max()  # 100
    df["age"] = df["age"].apply(lambda x: (x - age_min) / (age_max - age_min))


def handle_antiguedad(df):
    print("开始处理 antiguedad...")
    df['antiguedad'] = handle_nan(df['antiguedad'], 0)
    df['antiguedad'] = df['antiguedad'].astype(int)
    min_v = df['antiguedad'].min()
    max_v = df['antiguedad'].max()
    df['antiguedad'] = df['antiguedad'].apply(lambda x: (x - min_v) / (max_v - min_v))


def handle_renta(df):
    print("开始处理 renta...")
    mean_value = df['renta'].mean(skipna=True)
    df['renta'] = handle_nan(df['renta'], mean_value)
    min_value = df['renta'].min()
    max_value = df['renta'].max()
    df['renta'] = df['renta'].apply(lambda x: (x - min_value) / (max_value - min_value))



def handle_discrete_feature(df):
    print("开始处理 离散值...")
    for f in discrete_features:
        # 处理缺失值: NA '' NaN
        df[f] = handle_nan(df[f], "NAN")
        values = list(df[f].unique())
        mapping = {}
        # 特殊情况
        if f == 'indrel_1mes':
            mapping = {'NAN': 0, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5}
        else:
            for v in values:
                mapping[str(v)] = values.index(v)
        df[f] = df[f].map(lambda x: mapping.get(str(x)))
        df[f].astype(int)


def handle_prod(df):
    print("开始处理 product...")
    for p in products_list:
        df[p] = handle_nan(df[p], 0)
        df[p] = df[p].astype(int)


def handle_date(df):
    print("开始处理 date...")
    df_date = pd.to_datetime(df["fecha_dato"], format="%Y-%m-%d")
    df["month"] = pd.DatetimeIndex(df_date).month
    features_list.append("month")


def clean_train_data(file):
    # 读取数据
    t = datetime.datetime.now()
    df_orign = pd.read_csv(file)
    print("读取文件耗时: " + str(datetime.datetime.now() - t))

    print("开始进行数据预处理...")
    t = datetime.datetime.now()
    # 处理连续值，规格化
    handle_age(df_orign)
    handle_antiguedad(df_orign)
    handle_renta(df_orign)
    # 处理离散值
    handle_discrete_feature(df_orign)
    # 处理产品
    handle_prod(df_orign)
    # 处理日期
    handle_date(df_orign)
    print("数据预处理耗时: " + str(datetime.datetime.now() - t))
    return df_orign


def cut_df(df, dates):
    return df[df['fecha_dato'].isin(dates)]


def process_train_data(df, list_dates):
    print("开始构造训练数据...")
    t = datetime.datetime.now()
    dates = set()
    for date_pair in list_dates:
        dates.add(date_pair[0])
        dates.add(date_pair[1])
    df = cut_df(df, list(dates))
    train_list = []
    train_label = []
    user_ids = df['ncodpers'].unique()
    print(len(user_ids))
    num = 0
    for usr in user_ids:
        num += 1
        if num % 1000 == 0:
            print(num)
        for date in list_dates:
            prev_date = date[0]
            post_date = date[1]
            prev_products = df.loc[(df.fecha_dato == prev_date) & (df.ncodpers == usr), products_list].values.tolist()
            if len(prev_products) > 0:
                prev_products = prev_products[0]
            else:
                prev_products = [0] * len(products_list)
            post_products = df.loc[(df.fecha_dato == post_date) & (df.ncodpers == usr), products_list].values.tolist()
            if len(post_products) > 0:
                post_products = post_products[0]
            else:
                post_products = [0] * len(products_list)
            new_products = [max(x1 - x2, 0) for (x1, x2) in zip(post_products, prev_products)]
            if sum(new_products) > 0:
                for ind, prod in enumerate(new_products):
                    if prod > 0:
                        features = \
                        df.loc[(df.fecha_dato == prev_date) & (df.ncodpers == usr), features_list].values.tolist()[0]
                        train_list.append(features + prev_products)
                        train_label.append(ind)
    print("构造训练数据耗时: " + str(datetime.datetime.now() - t))
    return train_list, train_label


def clean_test_data(file):
    # 读取数据
    t = datetime.datetime.now()
    df_orign = pd.read_csv(file)
    print("读取文件耗时: " + str(datetime.datetime.now() - t))

    print("开始进行数据预处理...")
    t = datetime.datetime.now()
    # 处理连续值，规格化
    handle_age(df_orign)
    handle_antiguedad(df_orign)
    handle_renta(df_orign)
    # 处理离散值
    handle_discrete_feature(df_orign)
    # 处理日期
    handle_date(df_orign)
    print("数据预处理耗时: " + str(datetime.datetime.now() - t))
    return df_orign


def process_test_data(df, list_dates):
    print("开始构造测试数据...")
    t = datetime.datetime.now()
    test_list = []
    user_ids = df['ncodpers'].unique()
    for usr in user_ids:
        for date in list_dates:
            prev_products = df.loc[(df.fecha_dato == date) & (df.ncodpers == usr), products_list].values.tolist()
            if len(prev_products) > 0:
                prev_products = prev_products[0]
            else:
                prev_products = [0] * len(products_list)
        features = df.loc[(df.fecha_dato == '2016-06-28') & (df.ncodpers == usr), features_list].values.tolist()
        test_list.append(features + prev_products)
    print("构造测试数据耗时: " + str(datetime.datetime.now() - t))
    return test_list


def runXGB(train_X, train_y, seed_val=123):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 2
    param['subsample'] = 0.9
    param['colsample_bytree'] = 0.9
    param['seed'] = seed_val
    num_rounds = 110

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
    return model


def rec(train_file, test_file, res_file, list_dates=[('2015-05-28', '2015-06-28')]):
    train_df = clean_train_data(train_file)
    train_list, train_label = process_train_data(train_df, list_dates)
    # print(len(train_list), len(train_list[0]))
    test_df = clean_test_data(test_file)
    test_list = process_test_data(test_df, ['2016-05-28'])
    # print(len(test_list), len(test_list[0]))

    train_X = np.array(train_list)
    train_y = np.array(train_label)
    print(np.unique(train_y))
    print(train_X.shape, train_y.shape)

    test_X = np.array(test_list)
    print(test_X.shape)

    print("Building model..")
    model = runXGB(train_X, train_y, seed_val=0)
    del train_X, train_y
    print("Predicting..")
    xgtest = xgb.DMatrix(test_X)
    preds = model.predict(xgtest)
    del test_X, xgtest

    print("Getting the top products..")
    target_cols = np.array(products_list)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :8]
    test_id = np.array(pd.read_csv(test_file, usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv(res_file, index=False)


if __name__ == '__main__':
    train_file = "train_ver2.csv"
    test_file = 'test_ver2.csv'
    res_file = 'res.csv'
    rec(train_file, test_file, res_file)

    # s = np.NaN
    # print(type(s))
    # print(s is np.NaN)
