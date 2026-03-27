import sys
import os
# Le decimos a Python que agregue la carpeta padre (EmbedIA) a su radar de búsqueda
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importamos las herramientas de EmbedIA
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import ProjectOptions, ProjectType, ModelDataType

# 1. Carga de datos (Diabetes)

df = pd.read_csv(r"C:\Users\Usuario\Desktop\PPS\Datasets\diabetes.csv")
X = df.drop(columns=['Outcome']).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Entrenamiento (Scaler + LogReg)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

model.name = "modelo_diabetes"

# 3. Configuración de Exportación
options = ProjectOptions()
options.project_type = ProjectType.C
options.data_type = ModelDataType.FLOAT
# Agregamos el scaler como pre-procesamiento para que EmbedIA lo incluya
options.preprocessing = scaler

# 4. Generación del Proyecto
OUTPUT_FOLDER = r"C:/Users/Usuario\Desktop\PPS"
PROJECT_NAME = "diabetes_logistic_test"

generator = ProjectGenerator(options)
# Le pasamos el modelo de Scikit-Learn directamente
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print(f"¡Éxito! El proyecto se generó en la carpeta: {OUTPUT_FOLDER}/{PROJECT_NAME}")

# --- PRUEBA DE PREDICCIÓN ---
# Agarramos al paciente 0 del set de test (datos CRUDOS, sin escalar)
paciente = X_test[0]

# Formateamos el array para copiarlo fácil a C
paciente_c = ", ".join([f"{x}f" for x in paciente])
print(f"\nfloat raw_data[] = {{ {paciente_c} }};")

# Hacemos la predicción en Python
# OJO: A model.predict le pasamos el dato ESCALADO
paciente_escalado = scaler.transform([paciente])
prediccion_clase = model.predict(paciente_escalado)[0]

print(f"Predicción en Python (Clase): {prediccion_clase}")