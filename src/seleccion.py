"""Selección final PSet3: costos hipotéticos y reconstrucción de params.
No usa test para umbrales ni selección de features.
"""
from __future__ import annotations
import ast
import json
import numpy as np
from sklearn.metrics import confusion_matrix


def parsear_parametros(data):
    """Convierte el JSON de parámetros persistido por el notebook 04.
    'None' -> None; 'balanced' se conserva como string.
    """
    raw = json.loads(data) if isinstance(data, str) else dict(data)
    result = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                result[k] = ast.literal_eval(v)
            except (SyntaxError, ValueError):
                result[k] = v
        else:
            result[k] = v
    return result


def puntaje_costo(y_real, probabilidades, umbral, *, costo_contacto=5.,
                  costo_abandono=120., eficacia_hipotetica=0.20):
    """Costo *esperado bajo supuestos*, normalizado por 100 clientes.

    Todos los abandonos verdaderos sin intervención son pérdidas potenciales;
    entre TP contactados, hipotéticamente se retiene una proporción 'eficacia'.
    El costo de contacto aplica a TP y FP (todos los seleccionados).
    """
    etiquetas = np.asarray(y_real).astype(int)
    scores = np.asarray(probabilidades).astype(float)
    if len(etiquetas) != len(scores) or not len(etiquetas):
        raise ValueError('y y scores deben tener igual longitud no vacía')
    if not 0 <= eficacia_hipotetica <= 1:
        raise ValueError('Eficacia hipotética fuera de [0,1]')
    pred = (scores >= float(umbral)).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(etiquetas, pred, labels=[0, 1]).ravel())
    n_contactos = tp + fp
    costo = n_contactos*costo_contacto + (fn + tp*(1-eficacia_hipotetica))*costo_abandono
    return {
        'threshold': float(umbral), 'n': len(etiquetas), 'tp': tp, 'fp': fp,
        'fn': fn, 'tn': tn, 'contactados': n_contactos,
        'cost_per100': float(100*costo/len(etiquetas)),
        'cost_sin_contactos_per100': float(100*etiquetas.sum()*costo_abandono/len(etiquetas)),
        'ahorro_ilustrativo_per100': float(100*(etiquetas.sum()*costo_abandono-costo)/len(etiquetas)),
    }


def elegir_umbral_oof(y_real, probabilidades, *, costo_contacto=5.,
                        costo_abandono=120., eficacia_hipotetica=.20):
    # Malla prefijada antes de mirar test; ningún ajuste sobre hold-out.
    costos = [puntaje_costo(y_real, probabilidades, t,
                           costo_contacto=costo_contacto,
                           costo_abandono=costo_abandono,
                           eficacia_hipotetica=eficacia_hipotetica)
              for t in np.round(np.arange(.05, .951, .01), 2)]
    # Desempate: umbral mayor para reducir contactos cuando costos empatan.
    ganador = sorted(costos, key=lambda c: (c['cost_per100'], -c['threshold']))[0]
    return ganador, costos
