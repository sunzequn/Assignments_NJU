import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

# 去除了'tipodom'特征因为取值都是1
# 去除了'ult_fec_cli_1t' 'fecha_alta'
# 'fecha_dato'与'ncodpers'单独处理

# features_list = ['ind_empleado', 'pais_residencia', 'sexo', 'age',
#                  'ind_nuevo', 'antiguedad', 'indrel','indrel_1mes',
#                  'tiprel_1mes', 'indresi', 'indext', 'conyuemp',
#                  'canal_entrada', 'indfall', 'cod_prov', 'nomprov',
#                  'ind_actividad_cliente', 'renta', 'segmento']

# 去了省份相关特征
# features_list = ['ind_empleado', 'pais_residencia', 'sexo', 'age',
#                  'ind_nuevo', 'antiguedad', 'indrel','indrel_1mes',
#                  'tiprel_1mes', 'indresi', 'indext', 'conyuemp',
#                  'canal_entrada', 'indfall',
#                  'ind_actividad_cliente', 'renta', 'segmento']

# 最少特征
features_list = ['ind_empleado', 'pais_residencia', 'sexo', 'age',
                 'ind_nuevo', 'antiguedad',
                 # 'tiprel_1mes', 'indresi', 'indext',
                 'canal_entrada', 'nomprov',
                 'ind_actividad_cliente', 'renta', 'segmento']

# 去掉了'ind_ahor_fin_ult1', 'ind_aval_fin_ult1',
products_list = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

continuous_features = ['age', 'antiguedad', 'renta']
discrete_features = list(set(features_list) ^ set(continuous_features))


def handle_nan(df_col, v, is_strip=False):
    if is_strip:
        return df_col.map(lambda x: v if pd.isnull(x) or str(x).strip() == 'NA' or str(x).strip() == '' else str(x).strip())
    else:
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
    df["age"] = df["age"].map(lambda x: (x - age_min) / (age_max - age_min))


def handle_antiguedad(df):
    print("开始处理 antiguedad...")
    df['antiguedad'] = handle_nan(df['antiguedad'], 0)
    df["antiguedad"] = df["antiguedad"].map(lambda x: int(x.strip()) if (isinstance(x, str)) else x)
    df['antiguedad'] = df['antiguedad'].astype(int)
    min_v = df['antiguedad'].min()
    max_v = df['antiguedad'].max()
    df['antiguedad'] = df['antiguedad'].map(lambda x: (x - min_v) / (max_v - min_v))


def handle_renta(df):
    print("开始处理 renta...")
    # mean_value = df['renta'].mean(skipna=True)
    df['renta'] = handle_nan(df['renta'], 101850)
    df["renta"] = df["renta"].map(lambda x: x.strip() if (isinstance(x, str)) else x)
    df['renta'] = df['renta'].astype(float)
    min_value = df['renta'].min()
    max_value = df['renta'].max()
    df['renta'] = df['renta'].map(lambda x: (x - min_value) / (max_value - min_value))


def handle_discrete_feature(df):
    print("开始处理 离散值...")
    f_mapping = {}
    for f in discrete_features:
        # 处理缺失值: NA '' NaN
        df[f] = handle_nan(df[f], "SUNZEQUN", is_strip=True)
        values = list(df[f].unique())
        mapping = {}
        # 特殊情况
        if f == 'indrel_1mes':
            mapping = {'SUNZEQUN': 0, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5}
        else:
            for v in values:
                mapping[str(v)] = values.index(v)
        f_mapping[f] = mapping
        print(f, mapping)
        df[f] = df[f].map(lambda x: mapping.get(str(x)))
        df[f].astype(int)
    return df, f_mapping


def handle_discrete_feature_test(df, mappings):
    print("开始处理 离散值...")
    for f in discrete_features:
        # 处理缺失值: NA '' NaN
        df[f] = handle_nan(df[f], "SUNZEQUN", is_strip=True)
        mapping = mappings[f]
        df[f] = df[f].map(lambda x: mapping.get(str(x), -1))
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
    df_orign, mappings = handle_discrete_feature(df_orign)
    # 处理产品
    handle_prod(df_orign)
    # 处理日期
    handle_date(df_orign)
    print("数据预处理耗时: " + str(datetime.datetime.now() - t))
    return df_orign, mappings


def cut_df(df, dates):
    return df[df['fecha_dato'].isin(dates)]


def process_train_data(df, list_dates):
    print("开始构造训练数据...")
    prev_dates = list_dates[0]
    post_dates = list_dates[1]
    t = datetime.datetime.now()
    dates = set()
    for date in prev_dates:
        dates.add(date)
    for date in post_dates:
        dates.add(date)
    df = cut_df(df, list(dates))
    train_list = []
    train_label = []

    num = 0
    f = True
    user_products_dict = {}

    for index, row in df.iterrows():
        num += 1
        if num % 100000 == 0:
            print(num)
        usr = int(row['ncodpers'])

        if row['fecha_dato'] in post_dates:
            prev_products = user_products_dict.get(usr, [0] * len(products_list))
            post_products = row[products_list].values.tolist()
            new_products = [max(x1 - x2, 0) for (x1, x2) in zip(post_products, prev_products)]
            if sum(new_products) > 0:
                for ind, prod in enumerate(new_products):
                    if prod > 0:
                        features = row[features_list + ['month']].values.tolist()
                        if f:
                            print(len(features), len(prev_products))
                            f = False
                        train_list.append(features + prev_products)
                        train_label.append(ind)

        if row['fecha_dato'] in prev_dates:
            user_products_dict[usr] = row[products_list].values.tolist()

    print("构造训练数据耗时: " + str(datetime.datetime.now() - t))
    return train_list, train_label, user_products_dict


def clean_test_data(file, mappings):
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
    handle_discrete_feature_test(df_orign, mappings)
    # 处理日期
    handle_date(df_orign)
    print("数据预处理耗时: " + str(datetime.datetime.now() - t))
    return df_orign


def process_test_data(df, user_products_dict):
    print("开始构造测试数据...")
    t = datetime.datetime.now()
    test_list = []
    num = 0
    f = True
    for index, row in df.iterrows():
        num += 1
        if num % 200000 == 0:
            print(num)
        usr = int(row['ncodpers'])
        prev_products = user_products_dict.get(usr, [0] * len(products_list))
        features = row[features_list + ['month']].values.tolist()
        if f:
            print(len(features), len(prev_products))
            f = False
        test_list.append(features + prev_products)
    print(num)
    print("构造测试数据耗时: " + str(datetime.datetime.now() - t))

    return test_list


def xgb_model(train_X, train_y, seed_val=0):
    param = {'objective': 'multi:softprob', 'eta': 0.05, 'max_depth': 6, 'silent': 1, 'num_class': 22,
             'eval_metric': "mlogloss", 'min_child_weight': 2, 'subsample': 0.9, 'colsample_bytree': 0.9,
             'seed': seed_val}
    num_rounds = 100
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
    return model


def rec(train_file, test_file, res_file, list_dates=[['2015-05-28', '2016-05-28'], ['2015-06-28', '2016-06-28']]):
    train_df, mappings = clean_train_data(train_file)
    train_list, train_label, user_products_dict = process_train_data(train_df, list_dates)
    test_df = clean_test_data(test_file, mappings)
    test_list = process_test_data(test_df, user_products_dict)

    train_X = np.array(train_list)
    train_y = np.array(train_label)
    print(np.unique(train_y))
    print(train_X.shape, train_y.shape)

    test_X = np.array(test_list)
    print(test_X.shape)

    print("Building model..")
    model = xgb_model(train_X, train_y, seed_val=0)
    del train_X, train_y
    print("Predicting..")
    xgtest = xgb.DMatrix(test_X)
    preds = model.predict(xgtest)
    del test_X, xgtest

    print("Getting the top products..")
    target_cols = np.array(products_list)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :7]
    test_id = np.array(pd.read_csv(test_file, usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv(res_file, index=False)


if __name__ == '__main__':
    train_file = "train_ver2.csv"
    test_file = 'test_ver2.csv'
    res_file = 'res_quick_fix.csv'
    pre_dates = ['2015-05-28', '2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']
    post_dates = ['2015-06-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28', '2016-06-28']
    dates = [pre_dates, post_dates]
    t = datetime.datetime.now()
    rec(train_file, test_file, res_file)
    print("总耗时: ", datetime.datetime.now() - t)


