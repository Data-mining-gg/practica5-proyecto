from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


# RUTAS DE DATA

RUTA_DATA_RAW = RAIZ / "data" / "data-raw"

RUTA_DATA_PROCESSED = RAIZ / "data" / "data-processed"

RUTA_DATA_MODEL = RAIZ / "data" / "data-model"

# RUTAS DE SRC

RUTA_SRC_CLEANING = RAIZ / "src" / "cleaning"

RUTA_SRC_MODELS = RAIZ / "src" / "models"

RUTA_SRC_VISUAL = RAIZ / "src" / "visualization"