#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:16:40 2022

@author: rramirez
"""
import json

from flask import Flask, render_template, request
from hashlib import blake2b

from main_tests import Procesar
import main_tests as mt

app = Flask(__name__)

@app.route('/')
def index():
    datos, columnas_stats, datosStats ,kmo_model, ev, cm, inertias, km_cm, cmAD = Procesar()
    estadisticos=[idx.replace('%', '') for idx in datosStats.index]
    vals_estadisticos = dict()
    for col in columnas_stats:
        vals_estadisticos[col] = dict();
        for estad, lbl in zip(list(datosStats.index), estadisticos):
            vals_estadisticos[col][lbl] = f'{datosStats.loc[estad, col]:0.3f}'
    return render_template(
        'index.html',
        columnas_stats=json.dumps(columnas_stats),
        estadisticos=json.dumps(estadisticos),
        vals_estadisticos = json.dumps(vals_estadisticos),
        kmo_model=kmo_model,
        ev=ev,
        cm=cm,
        inertias=inertias,
        km_cm=km_cm,
        cmAD=cmAD
        )

def verificaPWD(pwd):
    h = blake2b(digest_size=20)
    h.update(pwd.encode('utf-8'))
    txtsha = h.hexdigest()
    return '6bfd2b5ef181219fad1f9b5356f9d43f4ac485b9' == txtsha or True


@app.route('/crear-modelos/', methods = ['POST', 'GET'])
def crear_modelos():
    if request.method == 'POST' and request.form['continue'] == "true":
        if verificaPWD(request.form['pwd'].lower()):
            mt.CrearModelos()
            return render_template(
                'basic/base.html',
                mensajes = [{
                    'tipo': 'success',
                    'mensaje': 'Modelos creados correctamente'}]
            )
        else:
            return render_template(
                'crear.html', titulo="Crear Modelos",
                mensajes = [{
                    'tipo': 'danger',
                    'mensaje': 'contraseña incorrecta'
                }])
    return render_template('crear.html', titulo="Crear Modelos")

@app.route('/crear-graficos/', methods = ['POST', 'GET'])
def crear_graficos():
    if request.method == 'POST' and request.form['continue'] == "true":
        if verificaPWD(request.form['pwd'].lower()):
            mt.CrearGraficos()
            return render_template(
                'basic/base.html',
                mensajes = [{
                    'tipo': 'success',
                    'mensaje': 'Gráficos creados correctamente'}]
            )
        else:
            return render_template(
                'crear.html', titulo="Crear Gráficos",
                mensajes = [{
                    'tipo': 'danger',
                    'mensaje': 'contraseña incorrecta'
                }])
    return render_template('crear.html', titulo="Crear Gráficos")

@app.route('/crear-reportes/', methods = ['POST', 'GET'])
def crear_reportes():
    if request.method == 'POST' and request.form['continue'] == "true":
        if verificaPWD(request.form['pwd'].lower()):
            mt.CrearReportes()
            return render_template(
                'basic/base.html',
                mensajes = [{
                    'tipo': 'success',
                    'mensaje': 'Reportes creados correctamente'}]
            )
        else:
            return render_template(
                'crear.html', titulo="Crear Reportes",
                mensajes = [{
                    'tipo': 'danger',
                    'mensaje': 'contraseña incorrecta'
                }])
    return render_template('crear.html', titulo="Crear Reportes")

if "__main__" == __name__:
    app.run()
