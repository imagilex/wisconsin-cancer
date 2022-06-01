#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 19:29:45 2022

@author: rramirez
"""
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

import funciones as F

DATA_TRAIN = 'data/CANCER.csv'
DATA_TEST = 'data/CANCER.csv'

MODELO_KMEANS = 'modelos/kmeans'
MODELO_LDALR = 'modelos/ldalr'
MODELO_FA = 'modelos/fa'
GRAFICOS = 'static/graficos/'
REPORTE_PRINCIPAL = 'reportes/principal'

intercambiarGruposKMeans = False

def CrearModelos():
    dataFrame, columnas_stats = CargarDataFrame()
    F.CrearModeloKMeans(
        dataFrame[columnas_stats], 2, MODELO_KMEANS)
    F.CrearModeloLDALR(
        dataFrame[columnas_stats], dataFrame['diagnosis_2'], 1, MODELO_LDALR)
    F.CrearModeloAF(
        dataFrame[columnas_stats], dataFrame['diagnosis_2'], 6, MODELO_FA)


def CrearGraficos():
    dataFrame, columnas_stats = CargarDataFrame()
    for col in columnas_stats:
        F.GeneraGraficoCaja(dataFrame[[col]], f'{GRAFICOS}/caja_{col}', col, [col])
        F.GeneraGraficoHistograma(dataFrame[[col]], f'{GRAFICOS}/hist_{col}', col, True)
        for colY in columnas_stats:
            F.GenerarGraficoDispersion(
                dataFrame[[col]],
                dataFrame[[colY]],
                f'{GRAFICOS}/disp_{col}_{colY}',
                f'{col} vs {colY}',
                col,
                colY,
                [
                    '#dc3545' if p == 1 else '#198754' 
                    for p in dataFrame[['diagnosis_2']].values[:,0]
                ])

    df = dataFrame.drop(columns = ['diagnosis'])
    modeloFA = FactorAnalyzer()
    modeloFA.set_params(n_factors=6, rotation="varimax")
    modeloFA.fit(df)
    eigenvalores, _ = modeloFA.get_eigenvalues()
    plt.scatter(range( 1, df.shape[1] + 1), eigenvalores)
    plt.plot(range( 1, df.shape[1] + 1), eigenvalores)
    plt.title('Factores Significativos')
    try:
        plt.xlabel('Factores')
    except:
        pass
    try:
        plt.ylabel('Eigenvalor')
    except:
        pass
    plt.grid()
    plt.savefig(f'{GRAFICOS}/Prueba_del_codo.png')
    plt.clf()

    ks = range(1, 10)
    inertias = []
    for k in ks:
        modeloKM=KMeans(n_clusters=k)
        modeloKM.fit(dataFrame[columnas_stats])
        inertias.append(modeloKM.inertia_)
    plt.plot(ks, inertias, '-o')
    plt.xlabel = 'Numero de clusters, K'
    plt.ylabel = 'Inertia'
    plt.xticks(ks)
    plt.savefig(f"{GRAFICOS}/km-inertia.png")
    plt.clf()

    modeloKM = F.CargarModelo(MODELO_KMEANS)
    predicciones = modeloKM.predict(dataFrame[columnas_stats])
    if intercambiarGruposKMeans:
        predicciones = [0 if p == 1 else 1 for p in predicciones]
    for col in columnas_stats:
        for colY in columnas_stats:
            F.GenerarGraficoDispersion(
                dataFrame[[col]], dataFrame[[colY]],
                f'{GRAFICOS}/km-disp_{col}_{colY}', f'{col} vs {colY}',
                col, colY, 
                ['#dc3545' if p == 1 else '#198754' for p in predicciones])

def CargarDataFrame():
    df = F.CargarDatos(DATA_TRAIN)
    df = F.LimpiarDatos(
        df, columnas_codificar=['diagnosis'], 
        columnas_eliminar=['id', 'Unnamed: 32'], eliminar_faltantes=True)
    
    cols_stats = list(df.columns)
    cols_stats.remove('diagnosis')
    cols_stats.remove('diagnosis_2')
    return (df, cols_stats)

def ReportePrincipal():
    datos, columnas_stats = CargarDataFrame()
    
    datosStats = datos[columnas_stats].describe()

    kmo_model,ev,cm = F.AnalisisFactorial(
        datos.drop(columns = ['diagnosis']),
        columnas_stats, 'diagnosis_2', MODELO_FA)

    cmAD = F.AnalisisDiscriminanteLineal(
        datos[columnas_stats], datos['diagnosis_2'], MODELO_LDALR)

    inertias, km_cm = F.ModeloKMeans(
        datos, columnas_stats, 'diagnosis_2', MODELO_KMEANS,
        intercambiarGruposKMeans)

    resultados = (
        datos,
        columnas_stats,
        datosStats,
        kmo_model,
        ev,
        cm,
        inertias,
        km_cm,
        cmAD
    )

    with open(REPORTE_PRINCIPAL, 'wb') as f:
        pickle.dump(resultados, f)

    return resultados

def CrearReportes():
    ReportePrincipal()

def Procesar():
    return F.CargarModelo(REPORTE_PRINCIPAL)

if "__main__" == __name__:
    CrearGraficos()
