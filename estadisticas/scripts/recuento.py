import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def contar_preguntas_y_modulos():
    BASE_DIR = Path(__file__).parent.parent.parent

    PREGUNTAS_PATH = BASE_DIR / "Preguntas.json"
    TIEMPOS_PATH = BASE_DIR / "evaluacion" / "json" / "tiempos.json"
    OUTPUT_TEX = BASE_DIR / "estadisticas" / "latex" / "tabla_preguntas_modulos.tex"
    OUTPUT_IMG = BASE_DIR / "estadisticas" / "graficos" / "modulos_frecuencia.png"

    # Cargar preguntas
    if not PREGUNTAS_PATH.exists():
        print(f"No se encontró Preguntas.json en: {PREGUNTAS_PATH}")
        return
    with open(PREGUNTAS_PATH, "r", encoding="utf-8") as f:
        preguntas_data = json.load(f)

    total_preguntas = sum(len(lista) for lista in preguntas_data.values())

    # Cargar tiempos "semi-JSON"
    if not TIEMPOS_PATH.exists():
        print(f"No se encontró tiempos.json en: {TIEMPOS_PATH}")
        return

    with open(TIEMPOS_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.endswith(","):
        content = content[:-1]
    content = "[" + content + "]"
    tiempos_data = json.loads(content)

    modulo_counter = Counter()
    for entrada in tiempos_data:
        modulos = entrada.get("modulos", [])
        for mod in modulos:
            nombre = mod.get("modulo")
            if nombre:
                modulo_counter[nombre] += 1

    # Crear gráfico
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    nombres = list(modulo_counter.keys())
    valores = list(modulo_counter.values())
    plt.figure(figsize=(10, 5))
    plt.bar(nombres, valores, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Frecuencia de activaciones por módulo")
    plt.ylabel("Número de activaciones")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    plt.close()

    # Documento LaTeX completo
    relative_img_path = Path("..") / "graficos" / OUTPUT_IMG.name

    latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}

\section*{Recuento de Preguntas y Módulos}
\begin{table}[h]
\centering
\begin{tabular}{lc}
\toprule
\textbf{Categoría} & \textbf{Recuento} \\
\midrule
""" + f"Total preguntas & {total_preguntas} \\\\\n"

    for modulo, count in modulo_counter.items():
        modulo_escapado = modulo.replace("_", r"\_")
        latex += f"{modulo_escapado} & {count} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Recuento de preguntas y número de activaciones por módulo}
\label{tab:recuento_preguntas_modulos}
\end{table}

\section*{Gráfico de Activaciones por Módulo}
\begin{center}
\includegraphics[width=0.9\textwidth]{""" + relative_img_path.as_posix() + r"""}
\end{center}

\end{document}
"""

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Archivo LaTeX generado en: {OUTPUT_TEX}")
    print(f"Gráfico guardado en: {OUTPUT_IMG}")

if __name__ == "__main__":
    contar_preguntas_y_modulos()
