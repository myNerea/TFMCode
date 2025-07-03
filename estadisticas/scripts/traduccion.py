import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Definir ruta base (ajusta según tu sistema)
BASE_DIR = Path(__file__).parent.parent.parent

# Rutas ajustadas
input_path = BASE_DIR / "evaluacion" / "json" / "traduccion.json"
output_dir_tex = BASE_DIR / "estadisticas" / "latex"
output_dir_tex.mkdir(parents=True, exist_ok=True)

output_dir_imgs = BASE_DIR / "estadisticas" / "graficos"
output_dir_imgs.mkdir(parents=True, exist_ok=True)

output_tex = output_dir_tex / "traduccion_scores.tex"
img_preg_path = output_dir_imgs / "scatter_pregunta.png"
img_resp_path = output_dir_imgs / "scatter_respuesta.png"

# Leer JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def contar_palabras(texto):
    if not texto:
        return 0
    return len(texto.split())

# Datos para pregunta
score_pregunta = []
num_palabras_pregunta = []

# Datos para respuesta
score_respuesta = []
num_palabras_respuesta = []

for d in data:
    if "score_pregunta" in d and d.get("pregunta"):
        score_pregunta.append(d["score_pregunta"])
        num_palabras_pregunta.append(contar_palabras(d["pregunta"]))

    if "score_respuesta" in d and d.get("respuesta"):
        score_respuesta.append(d["score_respuesta"])
        num_palabras_respuesta.append(contar_palabras(d["respuesta"]))

media_pregunta = np.mean(score_pregunta) if score_pregunta else float('nan')
std_pregunta = np.std(score_pregunta) if score_pregunta else float('nan')

media_respuesta = np.mean(score_respuesta) if score_respuesta else float('nan')
std_respuesta = np.std(score_respuesta) if score_respuesta else float('nan')

if len(score_pregunta) > 1:
    corr_preg, pval_preg = pearsonr(num_palabras_pregunta, score_pregunta)
else:
    corr_preg, pval_preg = float('nan'), float('nan')

if len(score_respuesta) > 1:
    corr_resp, pval_resp = pearsonr(num_palabras_respuesta, score_respuesta)
else:
    corr_resp, pval_resp = float('nan'), float('nan')

def graficar_dispersion(x, y, titulo, xlabel, ylabel, ruta_guardado):
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, color='blue')
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(ruta_guardado)
    plt.close()

# Generar gráficos y guardarlos
graficar_dispersion(num_palabras_pregunta, score_pregunta,
                    "Número de palabras vs score pregunta",
                    "Número de palabras (pregunta)",
                    "Score pregunta",
                    img_preg_path)

graficar_dispersion(num_palabras_respuesta, score_respuesta,
                    "Número de palabras vs score respuesta",
                    "Número de palabras (respuesta)",
                    "Score respuesta",
                    img_resp_path)

# Para la ruta en LaTeX, relativa al .tex que está en 'estadisticas'
# las imágenes están en '../evaluacion/graficos'
ruta_img_preg_latex = "../graficos/" + img_preg_path.name
ruta_img_resp_latex = "../graficos/" + img_resp_path.name

latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{float}
\geometry{margin=1in}
\begin{document}
\section*{Estadísticas de puntuación de traducción}

\begin{table}[H]
\centering
\caption{Media y desviación típica de los scores de traducción}
\begin{tabular}{lcc}
\toprule
\textbf{Elemento} & \textbf{Media} & \textbf{Desviación típica} \\
\midrule
Pregunta & %.4f & %.4f \\
Respuesta & %.4f & %.4f \\
\bottomrule
\end{tabular}
\end{table}

\bigskip

\begin{table}[H]
\centering
\caption{Correlación de Pearson entre número de palabras y score}
\begin{tabular}{lcc}
\toprule
\textbf{Elemento} & \textbf{Correlación (r)} & \textbf{p-valor} \\
\midrule
Pregunta & %.4f & %.4f \\
Respuesta & %.4f & %.4f \\
\bottomrule
\end{tabular}
\end{table}

\bigskip

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{%s}
\caption{Dispersión entre número de palabras y score para preguntas}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{%s}
\caption{Dispersión entre número de palabras y score para respuestas}
\end{figure}

\end{document}
""" % (media_pregunta, std_pregunta, media_respuesta, std_respuesta,
       corr_preg, pval_preg, corr_resp, pval_resp,
       ruta_img_preg_latex, ruta_img_resp_latex)

with open(output_tex, "w", encoding="utf-8") as f:
    f.write(latex)

print(f" Archivo LaTeX generado en: {output_tex}")
print(f"Gráficos guardados en: {img_preg_path}, {img_resp_path}")
print(f"Media score pregunta: {media_pregunta:.4f}, desviación típica: {std_pregunta:.4f}")
print(f"Media score respuesta: {media_respuesta:.4f}, desviación típica: {std_respuesta:.4f}")
print(f"Correlación pregunta (r): {corr_preg:.4f}, p-valor: {pval_preg:.4f}")
print(f"Correlación respuesta (r): {corr_resp:.4f}, p-valor: {pval_resp:.4f}")
