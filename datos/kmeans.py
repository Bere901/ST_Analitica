# %%
import numpy as np;  np.random.seed(42)
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox_orders_with_categories_partials_2017.csv")
dfp = df[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]

print(df)
# %%

df.describe()

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
#sns.scatterplot(data=df, x="total_items", y="order", hue=kmeans.labels_)
#plt.show()
# %%
df["clusters"] = kmeans.labels_ 
cluster0 = df[df.clusters == 0]
sns.boxplot(x="variable", y="value", data=pd.melt(cluster0[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]))
# %%

cluster1 = df[df.clusters == 1]
sns.boxplot(x="variable", y="value", data=pd.melt(cluster1[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]))
# %%
cluster2 = df[df.clusters == 2]
sns.boxplot(x="variable", y="value", data=pd.melt(cluster2[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]))

# %%

cluster3 = df[df.clusters == 3]
sns.boxplot(x="variable", y="value", data=pd.melt(cluster3[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]))

# %%
cluster4 = df[df.clusters == 4]
sns.boxplot(x="variable", y="value", data=pd.melt(cluster4[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]))
# %%

cluster5 = df[df.clusters == 5]
sns.boxplot(x="variable", y="value", data=pd.melt(cluster5[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]))
# %%
df = pd.DataFrame(data = np.random.random(size=(8,8)), columns = ['Food%', 'Fresh%', 'Drinks%', 'Home%', 'Beauty%', 'Health%', 'Baby%', 'Pets%'])

sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier()
tree.fit(df[['Food%', 'Fresh%', 'Drinks%', 'Home%', 'Beauty%', 'Health%', 'Baby%', 'Pets%']], kmeans.labels_)
print(export_text(tree, feature_names=['Food%', 'Fresh%', 'Drinks%', 'Home%', 'Beauty%', 'Health%', 'Baby%', 'Pets%']))