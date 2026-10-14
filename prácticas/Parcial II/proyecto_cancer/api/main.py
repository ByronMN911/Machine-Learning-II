import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Se carga el modelo exportado desde el notebook
with open(Path(__file__).parent / "modelo_mp.json", encoding="utf-8") as f:
    MODELO = json.load(f)

app = FastAPI(title="API Neurona McCulloch y Pitts")

# CORS permite que la app (o una página HTML) consuma la API desde otro origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Entrada(BaseModel):
    # Ejemplo: {"caracteristicas": {"mean radius": 17.99, "mean texture": 10.38}}
    caracteristicas: dict[str, float]


@app.get("/parametros")
def parametros():
    """Devuelve la lista de parámetros con su rango y valor típico."""
    return [
        {
            "nombre": nombre,
            "minimo": MODELO["rangos"][nombre][0],
            "maximo": MODELO["rangos"][nombre][1],
            "mediana": MODELO["medianas"][nombre],
        }
        for nombre in MODELO["features"]
    ]


@app.post("/predecir")
def predecir(entrada: Entrada):
    desconocidos = [n for n in entrada.caracteristicas if n not in MODELO["features"]]
    if desconocidos:
        raise HTTPException(status_code=422, detail=f"Parámetros desconocidos: {desconocidos}")

    # Los parámetros no enviados se completan con la mediana del entrenamiento
    valores = {**MODELO["medianas"], **entrada.caracteristicas}

    # Cada característica pasa a binario: 1 si está en la mitad baja del rango, 0 si no
    suma = sum(
        1 if valores[nombre] <= MODELO["cortes"][nombre] else 0
        for nombre in MODELO["features"]
    )

    benigno = suma >= MODELO["threshold"]
    return {
        "suma": suma,
        "threshold": MODELO["threshold"],
        "benigno": benigno,
        "diagnostico": "Benigno (sin cáncer)" if benigno else "Maligno (posible cáncer)",
    }