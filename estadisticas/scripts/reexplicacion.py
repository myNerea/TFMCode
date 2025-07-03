import json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

# Rutas ajustadas
BASE_DIR = Path(__file__).parent.parent.parent
input_path = BASE_DIR / "evaluacion" / "json" / "reexplicaciones.json"
plot_dir = BASE_DIR / "estadisticas" / "graficos"
latex_path = BASE_DIR / "estadisticas" / "latex" / "reexplicacion_estadisticas.tex"

plot_dir.mkdir(parents=True, exist_ok=True)
latex_path.parent.mkdir(parents=True, exist_ok=True)

CAMPOS = ["fidelidad", "simplificacion", "alucinacion"]

def extraer_campos(respuesta_cruda):
    respuesta_cruda = respuesta_cruda.strip()
    if not respuesta_cruda:
        return {}
    start = respuesta_cruda.find("{")
    end = respuesta_cruda.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    fragmento = respuesta_cruda[start:end+1]
    try:
        data = json.loads(fragmento)
        return {k: data.get(k) for k in CAMPOS}
    except json.JSONDecodeError:
        return {}

# --- Leer JSON línea por línea ---
valores_campo = defaultdict(list)
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            entrada = json.loads(line)
            evaluacion = entrada.get("evaluacion", {})
            campos_extraidos = extraer_campos(evaluacion.get("respuesta_cruda", ""))
            for campo, valor in campos_extraidos.items():
                if valor is not None:
                    valores_campo[campo].append(str(valor))
        except json.JSONDecodeError as e:
            print(f"Error en línea JSON general: {e}\n{line}")

# --- Calcular moda por campo ---
modas = {}
for campo in CAMPOS:
    if valores_campo[campo]:
        c = Counter(valores_campo[campo])
        modas[campo] = c.most_common(1)[0][0]
    else:
        modas[campo] = "N/A"

# --- Mostrar resumen ---
print("Moda por campo:")
for campo in CAMPOS:
    print(f" - {campo}: {modas[campo]}")

# --- Crear gráficos individuales por campo ---
rutas_imgs_latex = {}
for campo in CAMPOS:
    c = Counter(valores_campo[campo])
    etiquetas = list(c.keys())
    cuentas = list(c.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(etiquetas, cuentas, color='lightcoral', edgecolor='black')
    ax.set_title(f"Frecuencia de valores en '{campo}'")
    ax.set_ylabel("Cantidad")
    ax.set_xlabel("Valor")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(cuentas):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom')

    plt.tight_layout()
    img_path = plot_dir / f"{campo}_frecuencias.png"
    plt.savefig(img_path)
    plt.close()

    rutas_imgs_latex[campo] = "../graficos/" + img_path.name
    print(f" Gráfico guardado en: {img_path}")

# --- Generar LaTeX ---
latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}

\section*{Moda de Evaluaciones en Reexplicaciones}

\begin{table}[H]
\centering
\caption{Valor más frecuente por campo}
\begin{tabularx}{0.7\textwidth}{lX}
\toprule
\textbf{Campo} & \textbf{Moda} \\
\midrule
"""

for campo in CAMPOS:
    valor_tex = modas[campo].replace("_", r"\_")
    latex += f"{campo.capitalize()} & {valor_tex} \\\\\n"

latex += r"""\bottomrule
\end{tabularx}
\end{table}
"""

# Insertar figuras por campo
for campo in CAMPOS:
    output_img = rutas_imgs_latex[campo]
    latex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{""" + output_img.replace("\\", "/") + r"""}
\caption{Frecuencia de valores para """ + campo.replace("_", " ") + r"""}
\end{figure}
"""

latex += r"\end{document}"

# --- Guardar LaTeX ---
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex)

print(f"\nArchivo LaTeX generado en: {latex_path}")
