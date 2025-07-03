import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

# Ajusta según dónde está tu script:
BASE_DIR = Path(__file__).parent.parent.parent

input_path = BASE_DIR / "evaluacion" / "json" / "no_rag.json"
output_plot = BASE_DIR / "estadisticas" / "graficos" / "no_rag_frecuencias.png"
output_tex = BASE_DIR / "estadisticas" / "latex" / "no_rag_estadisticas.tex"

# Crear directorios si no existen
output_plot.parent.mkdir(parents=True, exist_ok=True)
output_tex.parent.mkdir(parents=True, exist_ok=True)

# Leer JSON (manejo formato sin lista)
with open(input_path, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()
    if not raw_text.startswith("["):
        raw_text = "[" + raw_text
    if not raw_text.endswith("]"):
        raw_text = raw_text.rstrip(",") + "]"
    data = json.loads(raw_text)

campos = ["relevancia", "claridad", "coherencia"]
valores_campo = defaultdict(list)

for item in data:
    evaluacion = item.get("evaluacion", {})
    for campo in campos:
        valor = evaluacion.get(campo)
        if isinstance(valor, str):
            valor = valor.strip()
        if valor is not None:
            valores_campo[campo].append(valor)


modas = {}
for campo in campos:
    if valores_campo[campo]:
        c = Counter(valores_campo[campo])
        modas[campo] = c.most_common(1)[0][0]
    else:
        modas[campo] = None

print("Moda por campo:")
for campo, moda in modas.items():
    print(f" - {campo}: {moda}")

# Generar gráfico
fig, axs = plt.subplots(len(campos), 1, figsize=(8, 4 * len(campos)))
if len(campos) == 1:
    axs = [axs]

for ax, campo in zip(axs, campos):
    c = Counter(valores_campo[campo])
    etiquetas = list(c.keys())
    cuentas = list(c.values())
    ax.bar(etiquetas, cuentas, color='skyblue', edgecolor='black')
    ax.set_title(f"Frecuencia de valores en '{campo}'")
    ax.set_ylabel("Cantidad")
    ax.set_xlabel("Valor")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(cuentas):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(output_plot)
plt.close()

print(f"Gráfico guardado en: {output_plot}")

# Calcular ruta relativa de la imagen desde el directorio del .tex
relative_img_path = Path("..") / "graficos" / output_plot.name

latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}

\section*{Moda de evaluaciones no RAG}

\begin{table}[H]
\centering
\caption{Valor más frecuente (moda) por campo de evaluación}
\begin{tabularx}{0.6\textwidth}{lX}
\toprule
\textbf{Campo} & \textbf{Valor más frecuente} \\
\midrule
"""

for campo in campos:
    valor = modas[campo]
    valor_tex = valor.replace('_', '\\_') if valor is not None else "N/A"
    latex += f"{campo.capitalize()} & {valor_tex} \\\\\n"

latex += r"""\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{""" + str(relative_img_path).replace("\\", "/") + r"""}
\caption{Frecuencia de valores por campo de evaluación}
\end{figure}

\end{document}
"""

with open(output_tex, "w", encoding="utf-8") as f:
    f.write(latex)

print(f"\nArchivo LaTeX generado en: {output_tex}")
print("\n=== Código LaTeX generado ===\n")
print(latex)
