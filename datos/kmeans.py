# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox_orders_with_categories_partials_2017.csv")

dfp = df[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]

# %%

ssd = []
ks = range(1,11)
for k in range(1,11):
    km = KMeans(n_clusters=k)
    km = km.fit(dfp)
    ssd.append(km.inertia_)

kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()

k = round(kneedle.knee)

print(f"Number of clusters suggested by knee method: {k}")

# %%

kmeans = KMeans(n_clusters=k).fit(df[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]])
sns.scatterplot(data=df, x="Annual_Income_(k$)", y="Spending_Score", hue=kmeans.labels_)
plt.show()

##### quitar scatter solo trabajar box plot

cluster0 = df(KMeans.labels_ == 0)
cluster0.describe()
sns.boxplot(data = cluster0, x = "Annual_Income_(k$)")

#####

cluster1 = df(KMeans.labels_ = 1)
cluster1.describe()
sns.boxplot(data = cluster1, x = "Annual_Income_(k$)")

#####

df['cluster'] = KMeans.labels_
sns.boxplot(data = df, x = 'cluster', y = "Annual_Income_(k$)")

# %%

from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier()
tree.fit(df[["Age", "Annual_Income_(k$)", "Spending_Score"]], kmeans.labels_)
print(export_text(tree, feature_names=["Age", "Annual_Income_(k$)", "Spending_Score"]))