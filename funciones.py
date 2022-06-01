#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 19:29:13 2022

@author: rramirez
"""
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from factor_analyzer import FactorAnalyzer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def CargarDatos(archivo):
    if "csv" == archivo.split('.')[-1].lower():
        return pd.read_csv(archivo)
    elif "xlsx" == archivo.split('.')[-1].lower():
        return pd.read_excel(archivo, engine='openpyxl')
    
def CodificarColumna(dataFrame, columna):
    dataColCodificada = LabelEncoder().fit_transform(dataFrame[columna])
    dfColCodificada = pd.DataFrame(dataColCodificada, columns=[columna + '_2'])
    dfResultado = pd.merge(
        dataFrame, dfColCodificada, left_index=True, right_index=True)
    return dfResultado

def GenerarDummy(dataFrame, columna):
    df = pd.get_dummies(dataFrame[columna], prefix=columna)
    # Es necesario eliminar una columna para eliminar variables ficticias
    df.drop(columns=[df.columns[0]], inplace=True)
    dataFrame.drop(columns=[columna], inplace=True)
    return pd.merge(dataFrame, df, left_index=True, right_index=True)

def LimpiarDatos(
        dataFrame,
        columnas_codificar=None, columnas_eliminar=None, columnas_dummy=None,
        estandarizar=False, eliminar_faltantes=False):
    if columnas_eliminar and len(columnas_eliminar) > 0:
        dataFrame = dataFrame.drop(columns=columnas_eliminar)
    if columnas_codificar and len(columnas_codificar) > 0:
        for columna in columnas_codificar:
            dataFrame = CodificarColumna(dataFrame, columna)
    if columnas_dummy and len(columnas_dummy) > 0:
        for columna in columnas_dummy:
            dataFrame = GenerarDummy(dataFrame, columna)
    if eliminar_faltantes:
        dataFrame = dataFrame.dropna()
    if estandarizar:
        dataFrame = pd.DataFrame(
            StandardScaler().fit_transform(dataFrame),
            columns=dataFrame.columns)
    return dataFrame

def GeneraGraficoHistograma(dataFrame, archivo, titulo, con_acumulativa=False):
    res = plt.hist(dataFrame)
    if con_acumulativa:
        punto0 = (res[1][1] - res[1][0]) / 2
        suma = 0
        x = [res[1][0] - punto0]
        y = [suma]
        for idx, frec in enumerate(res[0]):
            suma += frec
            x.append((res[1][idx] + res[1][idx + 1]) / 2)
            y.append(suma)
        plt.plot(x,y)
        for px, py in zip(x, y):
            lbl = f"{py/suma*100:0.0f}%"
            plt.annotate(
                lbl, (px, py),
                textcoords="offset points", xytext=(0, 5), ha='center')
    plt.title(titulo)
    plt.savefig(f"{archivo}.png")
    plt.close('all')
    
def GeneraGraficoCaja(dataFrame, archivo, titulo, etiquetas):
    fig, ax = plt.subplots()
    ax.boxplot(dataFrame, labels=etiquetas)
    plt.title(titulo)
    plt.savefig(f"{archivo}.png")
    plt.close('all')

def GenerarGraficoDispersion(
        dataFrameX, dataFrameY, archivo, titulo, ejeX, ejeY, colores=None):
    if (not colores is None) and len(colores) > 0:
        plt.scatter(dataFrameX, dataFrameY, c=colores)
    else:
        plt.scatter(dataFrameX, dataFrameY)
    try:
        plt.xlabel(ejeX)
    except:
        pass
    try:
        plt.ylabel(ejeY)
    except:
        pass
    plt.title(titulo)
    plt.savefig(f"{archivo}.png")
    plt.close('all')

def AnalisisFactorial(df, cols_stats, colPred, ruta_modelo):
    _, kmo_model = calculate_kmo(df)
    fa = FactorAnalyzer()
    fa.set_params(n_factors=6, rotation="varimax")
    fa.fit(df)
    ev, _ = fa.get_eigenvalues()

    modelo = CargarModelo(ruta_modelo)
    predicciones = modelo.predict(X=df[cols_stats])
    predicciones = predicciones.flatten()
    cm = confusion_matrix(df[colPred], predicciones)
    return (kmo_model, ev, cm)

def AnalisisDiscriminanteLineal(df_indep, df_dep, ruta_modelo):
    modeloLDA = CargarModelo(ruta_modelo + '_lda')
    modeloLR = CargarModelo(ruta_modelo + "_lr")
    scaler = CargarModelo(ruta_modelo + "_scaler")

    df_indep = scaler.transform(df_indep)
    df_indep = modeloLDA.transform(df_indep)
    y_pred = modeloLR.predict(df_indep)
    cm = confusion_matrix(df_dep, y_pred)
    return cm
    
def ModeloKMeans(datos, columnas_stats, colPred, ruta_modelo, intercambiarGruposKMeans):

    ks = range(1, 5)
    inertias = []

    for k in ks:
        modeloKM=KMeans(n_clusters=k)
        modeloKM.fit(datos[columnas_stats])
        inertias.append(modeloKM.inertia_)

    modeloKM = CargarModelo(ruta_modelo)
    predicciones = modeloKM.predict(datos[columnas_stats])
    if intercambiarGruposKMeans:
        predicciones = [0 if p == 1 else 1 for p in predicciones]
    cm = confusion_matrix(datos[colPred], predicciones)
    return (inertias, cm)

def CargarModelo(archivo):
    return pickle.load(open(archivo, 'rb'))

def CrearModeloAF(df_indep, df_dep, factores=6, archivo=None):
    modelo = make_pipeline( FactorAnalyzer(), LogisticRegression())
    modelo.named_steps['factoranalyzer'].set_params(n_factors=factores, rotation="varimax")
    modelo.fit(X=df_indep, y=df_dep)

    if archivo:
        with open(archivo, 'wb') as f:
            pickle.dump(modelo, f)

    return modelo

def CrearModeloLDALR(df_indep, df_dep, componentes=1, archivo=None):
    scaler = StandardScaler()
    df_indep = scaler.fit_transform(df_indep)
    modeloLDA = LDA(n_components=componentes)
    df_indep = modeloLDA.fit_transform(
        df_indep, df_dep)
    modeloLR = LogisticRegression(random_state=0)
    modeloLR.fit(df_indep, df_dep)

    if archivo:
        with open(archivo + "_scaler", 'wb') as f:
            pickle.dump(scaler, f)
        with open(archivo + "_lda", 'wb') as f:
            pickle.dump(modeloLDA, f)
        with open(archivo + "_lr", 'wb') as f:
            pickle.dump(modeloLR, f)

    return (scaler, modeloLDA, modeloLR)

def CrearModeloKMeans(df, clusters=2, archivo=None):
    modelo = KMeans(n_clusters=clusters)
    modelo.fit(df)

    if archivo:
        with open(archivo, 'wb') as f:
            pickle.dump(modelo, f)

    return modelo
