import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent.parent

input_path = BASE_DIR / "evaluacion" / "json" / "tiempos.json"
output_tex = BASE_DIR / "estadisticas" / "latex" / "tabla_tiempos.tex"
output_img_dir = BASE_DIR / "estadisticas" / "graficos"
output_img_dir.mkdir(parents=True, exist_ok=True)
output_img = output_img_dir / "tiempos_por_modulo.png"

with open(input_path, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()
    if not raw_text.startswith("["):
        raw_text = "[" + raw_text
    if not raw_text.endswith("]"):
        raw_text = raw_text.rstrip(",") + "]"
    data = json.loads(raw_text)

# Total times en minutos
tiempos_totales = np.array([d["tiempo_total"] / 60 for d in data])

# Obtener todos los módulos únicos
modulos = set()
for d in data:
    for m in d["modulos"]:
        modulos.add(m["modulo"])
modulos = sorted(modulos)

# Crear matriz: filas = muestras, columnas = módulos, rellenar con nan
matriz_modulos = np.full((len(data), len(modulos)), np.nan)

# Rellenar matriz
modulo_idx = {m: i for i, m in enumerate(modulos)}
for i, d in enumerate(data):
    for m in d["modulos"]:
        idx = modulo_idx[m["modulo"]]
        matriz_modulos[i, idx] = m["duracion_segundos"] / 60

# Función para correlación ignorando nan
def corr_ignoring_nan(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]

# Calcular medias, std y correlaciones
medias = np.nanmean(matriz_modulos, axis=0)
stds = np.nanstd(matriz_modulos, axis=0)
correlaciones = [corr_ignoring_nan(matriz_modulos[:, i], tiempos_totales) for i in range(len(modulos))]

media_total = np.mean(tiempos_totales)
std_total = np.std(tiempos_totales)

# Generar LaTeX
latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
\section*{Estadísticas de tiempo por módulo y total}

\begin{table}[H]
\centering
\caption{Estadísticas de tiempo por módulo y total (minutos)}
\begin{tabularx}{\textwidth}{lRRR}
\toprule
\textbf{Módulo} & \textbf{Media} & \textbf{Desviación típica} & \textbf{Correlación con total} \\
\midrule
"""

for i, modulo in enumerate(modulos):
    media = medias[i]
    std = stds[i]
    corr = correlaciones[i]
    corr_str = "N/A" if np.isnan(corr) else f"{corr:.4f}"
    modulo_tex = modulo.replace('_', '\\_')
    latex += f"{modulo_tex} & {media:.4f} & {std:.4f} & {corr_str} \\\\\n"

latex += r"\midrule" + "\n"
latex += f"\\textbf{{Total}} & {media_total:.4f} & {std_total:.4f} & N/A \\\\\n"
latex += r"""\bottomrule
\end{tabularx}
\end{table}
"""

# Gráfico
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(modulos, medias, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
ax.set_ylabel("Tiempo medio (minutos)")
ax.set_title("Tiempo medio y desviación típica por módulo")
ax.set_xticks(range(len(modulos)))
ax.set_xticklabels(modulos, rotation=45, ha="right")
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_img)
plt.close()

latex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{""" + str(output_img).replace("\\", "/") + r"""}
\caption{Tiempo medio y desviación típica por módulo}
\end{figure}
"""

latex += r"\end{document}"

output_tex.parent.mkdir(parents=True, exist_ok=True)
with open(output_tex, "w", encoding="utf-8") as f:
    f.write(latex)

print(f" Archivo LaTeX generado en: {output_tex}")
print(f" Imagen guardada en: {output_img}")
