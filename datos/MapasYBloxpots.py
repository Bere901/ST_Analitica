#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("ulabox_orders_with_categories_partials_2017.csv")

#%% Tamaño del Dataframe
columnas = len(df.columns)
filas = len(df)
print(f"Número de columnas: {columnas}")
print(f"Número de filas: {filas}")


#%%Descripcion General del Dataframe
df.describe()

#grafica de cajas y bigotes
plt.boxplot(df.describe())
plt.title("Boxplot Using Matplotlib")
plt.show()

#%% Informacion sobre los tipos de datos
df.info()

#%%Ejemplo de los primeros 5 datos del Dataframe
df.head(5)

#%%Histograma de los departamentos (ordenes vs % de departamento)
departments = ("Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%")
for department in departments:
    plt.hist(df[[department]])
    plt.title(f"Histograma - {department}") 
    plt.ylabel("n de ordenes")
    plt.xlabel("% de compra")
    plt.show()
    
#%%Analisis del tiempo
df['weekday'].value_counts()
plt.hist(df[['weekday']])
plt.xlabel("Día de la semana")
plt.ylabel("n de ordenes registradas")
plt.show()

#%% Horas del día- Más ventas a menores ventas
df['hour'].value_counts()
plt.hist(df[['hour']])
plt.xlabel("Hora")
plt.ylabel("n de ordenes registradas")
plt.show()

#%%Intervalos de tiempo con más ordenes
days = list(range(1,8))
hours = list(range(0,24))

filler = np.zeros((len(days), len(hours)))

for i in range(len(days)):
    for j in range(len(hours)):
        val = len(df.loc[(df['hour'] == j + 1) & (df['weekday'] == i + 1)])
        filler[i][j] = val

heatDF = pd.DataFrame(filler, columns=hours, index=days)

plt.ylabel("Día de la semana")
plt.xlabel("Hora")
plt.title("Heatmap - Número de ordenes")
heatmap = plt.imshow(heatDF)
plt.colorbar(heatmap);

#%%
# df['discount%'] = df['discount%'].abs()
total_items = (df.loc[(df['discount%'] >= 0)])['total_items'].tolist()
discount = (df.loc[(df['discount%'] >= 0)])['discount%'].tolist()

plt.scatter(total_items, discount)
plt.xlabel("Numero de items comprados")
plt.ylabel("% de descuento")
plt.show()

#%%
days = list(range(1,8))
hours = list(range(0,24))
departments = ("Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%")
for department in departments:
    filler = np.zeros((len(days), len(hours)))

    for i in range(len(days)):
        for j in range(len(hours)):
            q = df.loc[(df['hour'] == j) & (df['weekday'] == i + 1)]
            data = q[department].tolist()
            avg = sum(data) / len(data)
            filler[i][j] = avg

    heatDF = pd.DataFrame(filler, columns=hours, index=days)

    plt.ylabel("Día de la semana")
    plt.xlabel("Hora")
    plt.title(f"Heatmap - Porcentaje promedio de\n ventas de {department[:-1]}")
    heatmap = plt.imshow(heatDF)
    plt.colorbar(heatmap)
    plt.show()
    
    