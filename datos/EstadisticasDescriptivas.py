# %%
import numpy as np 
import pandas as pd 

df = pd.read_csv('ulabox_orders_with_categories_partials_2017.csv')
print(df)

# %%
shape = df.shape
print('Filas: ', shape[0], 'Columnas: ', shape[1])
print()

#Tipos de datos 
print('Tipos de datos: ')
print(df.dtypes)

#%%
#Analisis de variables y rangos
print(''' Analisis de variables:
discount -> porcentaje de descuento
Food  -> porcentaje de comida
Fresh -> porcentaje de frescura de los alimentos
Drinks -> porcentaje de bebidas
Health -> porcentaje de salud\n ''')

#Valores maximos y minimos
print('Maximos y Minimos')
df.describe().round(2)

#%%
print('---Media---')
print(df.mean())
print('---Mediana---')
print(df.median())
print('---Moda---')
print(df.mode())
print('---Desviacion Estandar---')
print(df.std())
# %%
