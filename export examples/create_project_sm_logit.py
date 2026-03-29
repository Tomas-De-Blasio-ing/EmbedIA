import sys
import os
# Le decimos a Python que agregue la carpeta padre (EmbedIA) a su radar de búsqueda
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Importamos las herramientas de EmbedIA
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import ProjectOptions, ProjectType, ModelDataType

def load_dataset(filepath: str, target_col: str):
    '''
    Se carga el dataset y separa features de targets.
    Se retornoa X, y, nombre de los features.
    '''
    
    df = pd.read_csv(filepath)
    
    feature_names = [n for n in df.columns if n != target_col ] # Nos guardamos los nombres de las características

    if target_col not in df.columns:
        raise ValueError(
            f"Columna '{target_col}' no encontrada. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )
    
    X = df[feature_names].values
    y = df[target_col].values
    
    return X, y, feature_names

def train_model(X_data: np.ndarray ,y_data: np.ndarray):
    x_b = sm.add_constant(X_data) # Se agrega el bias a los datos escalados
    model = sm.Logit(y_data, x_b)
    res_model = model.fit(disp=0)
    return res_model

def search_best_aic(X: np.ndarray, y: np.ndarray, feature_names: list) -> tuple:
    '''
    Función para encontrar el mejor AIC del modelo.
    Se van eliminando uno a uno los features hasta encontrar el mejor AIC o que quede un feature.
    
    Se devuelve los índices de las caracterísitcas con mejor AIC
    
    '''
    
    activos = list(range(X.shape[1])) # Se guardan los índices de los features
    X_temp = X.copy()
    
    model_temp = train_model(X_temp,y)
    aic_temp = model_temp.aic
    
    mejora = True
    while mejora and len(activos) > 1:
        mejora = False # flag si mejoró el aic
        mejor_aic = aic_temp # guardo el mejor aic
        eliminar_indice = None # guardo el mejor conjunto con aic
        
        for indice in range(X_temp.shape[1]): # Por cada índice del conjunto actual
            X_candidato = np.delete(X_temp, indice, axis = 1) # Elimino el índice del conjunto
            try:                                                # Pruebo si mejoró el aic
                aic = train_model(X_candidato,y).aic
                if aic < mejor_aic:
                    mejor_aic = aic
                    eliminar_indice = indice
            except Exception:
                continue
        
        if eliminar_indice is not None: # Si el sistema tuvo una mejora, osea, un mejor conjunto
            eliminado = feature_names[activos[eliminar_indice]] # Imprimo los features a eliminar
            print(f"\nFeatures eliminados: {eliminado}")
            print(f"AIC: {aic_temp} --> {mejor_aic}\n")
            activos.pop(eliminar_indice)
            mejora = True
            X_temp = np.delete(X_temp, eliminar_indice, axis = 1)
            aic_temp = mejor_aic
    
    print(f"AIC final: {mejor_aic}, con {len(activos)} features activos")
    print(f"  Seleccionados: {[feature_names[i] for i in activos]}")

    return activos



# 1. Carga de datos
X, y, feature_names = load_dataset(r"C:\Users\Usuario\Desktop\PPS\Datasets\diabetes.csv", "Outcome")

# 2. Dividimos en entrenamiento y testeo
test_size: float = 0.2
random_state: int = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# 3. Normalización 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# 4. Selección de features por AIC
indices_features = search_best_aic(X_train_scaled,y_train,feature_names)

# 5. Re-entrenamos el scaler con las features seleccionadas del X ORIGINAL (sin escalar)
X_final = X_train[:, indices_features]
scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_final)


# 6. Modelo final
modelo_final = train_model(X_train_final_scaled,y_train)
modelo_final.name = "modelo_diabetes"
print(modelo_final.summary())


# 3. Configuración de Exportación
options = ProjectOptions()
options.project_type = ProjectType.C
options.data_type = ModelDataType.FLOAT
# Agregamos el scaler como pre-procesamiento para que EmbedIA lo incluya
options.preprocessing = scaler_final

# 4. Generación del Proyecto
OUTPUT_FOLDER = r"C:/Users/Usuario\Desktop\PPS"
PROJECT_NAME = "diabetes_logistic_test_statsmodels"

generator = ProjectGenerator(options)
# Le pasamos el modelo de Statsmodels directamente
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, modelo_final, options)

print(f"¡Éxito! El proyecto se generó en la carpeta: {OUTPUT_FOLDER}/{PROJECT_NAME}")

