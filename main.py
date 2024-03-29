import numpy as np
import pandas as pd
from html.parser import HTMLParser
import html
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

## TODO cosine without new users
#  just desc
#  dec and categ (handle categories differently ?)
#  dec and categ with NLP
#  trends for new users
#  new user cutoff search
#  trends also for old users
#  count add to carts ?


dfUsersPurchased = None
dfUsersAddedCart = None
userActionsCounts = None
dfProducts = None
dfUserActions = None
test = None
predictionsDF = None
train = None
topProducts = None
TRAIN_SPLIT = 0.6

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(htmlString):
    #there is some missing and or corrupt data so handle it properly
    try:
        #fuck you
        htmlString = html.unescape(htmlString)

        s = MLStripper()
        s.feed(htmlString)
    except:
        return ""
    return s.get_data()

def preProcess():
    global dfUsersPurchased
    global dfUsersAddedCart
    global dfUsersViewed
    global userActionsCounts
    global dfProducts
    global dfUserActions

    dfUserActions = pd.read_csv("C:\\Users\\igorf\\PycharmProjects\\vinf2\\data\\vi_dataset_events.csv")

    #remove view product events
    dfUserActions = dfUserActions[dfUserActions.type != 'view_product']

    timestamps = dfUserActions["timestamp"]
    dfUserActions = dfUserActions.drop(columns=['timestamp'])

    #separate events
    dfUsersPurchased = dfUserActions[dfUserActions.type == 'purchase_item']
    dfUsersAddedCart = dfUserActions[dfUserActions.type == 'add_to_cart']
    dfUsersViewed = dfUserActions[dfUserActions.type == 'view_product']

    #get number of actions for each user
    userActionsCounts = dfUserActions.groupby(["customer_id", "type"]).count()

    #rename column to avoid confusion
    userActionsCounts.columns = ['count']

    #usntack the hierarchical index created by grouping
    userActionsCounts = userActionsCounts.unstack()

    #replace nans
    userActionsCounts = userActionsCounts.fillna(0)

    dfUserActions = dfUserActions.join(timestamps)

    #TODO add back the timestamps of purchases for recent trend analysis

    dfProducts = pd.read_csv("C:\\Users\\igorf\\PycharmProjects\\vinf2\\data\\vi_dataset_catalog.csv")

    #drop all products that have not been purchased or added to cart
    #dfProducts = dfProducts[dfProducts['product_id'].isin(dfUserActions['product_id'])]



    #strip html from description
    dfProducts["description"] = dfProducts.apply(lambda row: strip_tags(row['description']), axis=1)

    cunts = dfProducts[dfProducts.duplicated(["description", "category_path", "brand", "gender", "price"], keep=False)].sort_values(by=["description", "category_path", "brand", "gender", "price"])

    prevRow = None
    for index, row in cunts.iterrows():
        if prevRow is not None:
            if row.iloc[2:].equals(prevRow.iloc[2:]):
                dfProducts.at[index, "product_id"] = prevRow["product_id"]
                dfUserActions["product_id"] = dfUserActions["product_id"].replace(row["product_id"], prevRow["product_id"])
        prevRow = row


    #!!BEWARE some descriptions are empty, some contain no usefull information, and couple are in slovak so we need co clear them next

def buildModel():
    global dfUsersPurchased
    global dfUsersAddedCart
    global dfUsersViewed
    global userActionsCounts
    global dfProducts
    global dfUserActions
    global test
    global train
    global predictionsDF
    global topProducts

    # drop all products that have not been purchased
    # dfProducts = dfProducts[dfProducts['product_id'].isin(dfUsersPurchased['product_id'])]
    #dfUsersPurchased = shuffle(dfUsersPurchased)
    dfUserActions = dfUserActions.sort_values(by=["timestamp"])
    split = int(dfUserActions.shape[0] * TRAIN_SPLIT)
    train, test = dfUserActions.iloc[:split], dfUserActions.iloc[split:]

    trainProd = dfProducts[dfProducts['product_id'].isin(train["product_id"])]
    trainPurchased = train[train.type == 'purchase_item']
    trainAddCart = train[train.type == 'add_to_cart']

    #get top sold products
    topProducts = dfUsersPurchased.groupby(["product_id"]).count()['customer_id'].sort_values(ascending=False)
    topProducts.columns = ["count"]

    #generate tfid matrix
    tf = TfidfVectorizer(strip_accents="unicode", analyzer="word", stop_words="english", ngram_range=(1,2), min_df=2, norm="l1")
    tfidMatrix = tf.fit_transform(trainProd["description"])

    tfidMatrix = tfidMatrix.toarray()
    tfidMatrix = np.transpose(tfidMatrix)

    #generate user profiles

    #split users

    #create user product matrix columns are users, rows are products
    trainPurchased["value"] = 1
    upTrainPurchased = pd.pivot_table(trainPurchased, values="value", index=["product_id"], columns="customer_id", fill_value=0)
    trainAddCart["value"] = 0
    upTrainAddCart = pd.pivot_table(trainAddCart, values="value", index=["product_id"], columns="customer_id", fill_value=0)

    updf = upTrainPurchased.add(upTrainAddCart, fill_value=0)
    updf = updf.fillna(0)

    min_max_scaler = preprocessing.MaxAbsScaler()
    x_scaled = min_max_scaler.fit_transform(updf.values)
    updf = pd.DataFrame(x_scaled, index=updf.index, columns=updf.columns.values)


    userProfiles = np.matmul(tfidMatrix, updf.values)
    #AT THIS POINT COLUMNS ARE USERS AND ROWS ARE TERMS
    #now we transpose the tfid matrix back and multiply again
    tfidMatrix = np.transpose(tfidMatrix)
    predictions = np.matmul(tfidMatrix, userProfiles)

    predictionsDF = pd.DataFrame(predictions, index=updf.index, columns=updf.columns.values)


def evalModel():
    global dfUsersPurchased
    global dfUsersAddedCart
    global userActionsCounts
    global dfProducts
    global dfUserActions
    global test
    global train
    global predictionsDF
    global topProducts

    test = test[test.type != 'add_to_cart']
    train = train[train.type == 'purchase_item']
    customers = pd.unique(test["customer_id"])

    trendHits = 0
    trendTotal = 0
    predHits = 0
    predTotal = 0

    for i in customers:
        numPurchased = train[train["customer_id"] == i].shape[0]
        reccomendations = None
        if numPurchased == 0:
            reccomendations = topProducts.head(5).index
            trendTotal = trendTotal + test[test["customer_id"] == i].shape[0]
            trendHits = trendHits + pd.Series(reccomendations).isin(test.loc[test["customer_id"] == i, ["product_id"]]["product_id"]).sum()
        else:
            reccomendations = predictionsDF[i].sort_values(ascending=False).head(5).index
            predTotal = predTotal + test[test["customer_id"] == i].shape[0]
            predHits = predHits + pd.Series(reccomendations).isin(test.loc[test["customer_id"] == i, ["product_id"]]["product_id"]).sum()

    print("Trend hit precision: " + str(trendHits/trendTotal) + " (" + str(trendHits) + " / " + str(trendTotal) + ")")
    print("Prediction hit precision: " + str(predHits/predTotal) + " (" + str(predHits) + " / " + str(predTotal) + ")")


preProcess()
buildModel()
evalModel()