from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import numpy as np
import pandas as pd
import math
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
# import japanize_matplotlib
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn import datasets

data_dict = {
    225: '日経平均株価',
    1963: '日揮ホールディングス',
    3407: '旭化成',
    3441: '山王',
    4080: '田中化学研究所',
    4088: 'エア・ウォーター',
    4091: '日本酸素ＨＤ',
    4208: '宇部興産',
    4406: '新日本理化',
    5019: '出光興産',
    5020: 'ＥＮＥＯＳ',
    5301: '東海カーボン',
    5631: '日本製鋼所',
    5659: '日本精線',
    5907: 'ＪＦＥコンテイナー',
    5922: '那須電機鉄工',
    5974: '中国工業',
    6269: '三井海洋開発',
    6330: '東洋エンジニアリング',
    6331: '三菱化工機',
    6391: '加地テック',
    6366: '千代田化工建設',
    6370: '栗田工業',
    6378: '木村化工機',
    6495: '宮入バルブ製作所',
    6498: 'キッツ',
    6752: 'パナソニック',
    6824: '新コスモス電機',
    6955: 'ＦＤＫ',
    7012: '川崎重工業',
    7013: 'ＩＨＩ',
    7203: 'トヨタ自動車',
    7246: 'プレス工業',
    7267: 'ホンダ',
    7715: '長野計器',
    7721: '東京計器',
    7727: 'オーバル',
    8015: '豊田通商',
    8088: '岩谷産業',
    8132: 'シナネン HD',
    8133: '伊藤忠エネクス',
}

flag_first = True
for mykey, company_name in data_dict.items():
    if mykey == 225:
        company_code = '^N225'
    else:
        company_code = str(mykey) + '.T'

    my_share = share.Share(company_code)

    try:
        # past 1 year data
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR, 5, share.FREQUENCY_TYPE_DAY, 1)

        headers = {}
        for key in symbol_data.keys():
            if key != 'timestamp':
                name = company_code + '_' + key
                headers[key] = name

        # transpose
        df = pd.DataFrame(symbol_data.values(), index=symbol_data.keys()).T
        # change column header
        df = df.rename(columns=headers)
        # convert data time
        df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
        # only need data part
        df.timestamp = df['timestamp'].dt.date

        print(company_code, company_name, len(df))

    except YahooFinanceError as e:
        print(e.message)

    if flag_first:
        df_all = df
        flag_first = False
    else:
        df_all = pd.merge(df_all, df)

df_all = df_all.dropna(how='any')
print(df_all)

df_value = df_all.iloc[:, 1:len(df_all.columns)]
print(df_value)
features = StandardScaler().fit_transform(df_value)
print(features)
print(features.shape)

# PCA
#pca = PCA()
pca = PCA(n_components=0.99, whiten=True)
features_pca = pca.fit_transform(features)
print(features_pca)
print('もとの特徴量数：', features.shape[1])
print('削減後の特徴量数：', features_pca.shape[1])

# Cumulative Contribution Ratio
#plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
#plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
#plt.xlabel("Number of Principal Components")
#plt.ylabel("Cumulative Contribution Ratio")
#plt.grid()
#plt.show()

features_train = features_pca[:len(features_pca)-1,:]
print(features_train.shape)
features_test = features_pca[len(features_pca)-1:,:]
print(features_test.shape)

tmp = df_all['6366.T_open']
#tmp = df_all['7715.T_open']
#tmp = df_all['8088.T_open']
#tmp = df_all['3441.T_open']
print(tmp)
print(len(tmp))
target_train = tmp[1:]
print(target_train)
print(len(target_train))

# Random Forest
randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)
model_rf = randomforest.fit(features_train, target_train)
pred = model_rf.predict(features_test)[0]
print(pred, 'Random Forest')

# PLS Regression
pls = PLSRegression(n_components=20, max_iter=100000)
model_pls = pls.fit(features_train, target_train)
pred = model_pls.predict(features_test)[0][0]
print(pred, 'PLS Regression')

# Bayesian Ridge Regression.
brr = linear_model.BayesianRidge()
model_brr = brr.fit(features_train, target_train)
pred = model_brr.predict(features_test)[0]
print(pred, 'Bayesian Ridge Regression')
