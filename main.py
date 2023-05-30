import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


df = pd.read_csv("FullData.csv")

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)

names = df['Name'].tolist()
drop_columns = ['Name', 'Nationality', 'Club',
       'Club_Position', 'Club_Kit', 'Club_Joining', 'Contract_Expiry', 'Preffered_Foot', 'Birth_Date', 'National_Position', 'National_Kit', 'Height', 'Weight']

df = df.drop(columns=drop_columns)

one_hot = pd.get_dummies(df['Preffered_Position'])
# Drop column B as it is now encoded
df = df.drop('Preffered_Position',axis = 1)
# Join the encoded df
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Work_Rate'])
# Drop column B as it is now encoded
df = df.drop('Work_Rate',axis = 1)
# Join the encoded df
df = df.join(one_hot)

x = df
y = names

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PCA1', 'PCA2', 'PCA3'])

finalDf = pd.concat([principalDf, pd.Series(names)], axis = 1)
#finalDf.to_csv("standardpca.csv")


kmeans_pca = KMeans(n_clusters=6, init="k-means++", random_state=42)
kmeans_pca.fit(principalComponents)

finalDf['Cluster'] = kmeans_pca.labels_
finalDf.to_csv("Final.csv")