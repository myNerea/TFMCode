import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import statistics

def analizar_rag(path=None):
    BASE_DIR = Path(__file__).parent.parent.parent

    input_path = BASE_DIR / "evaluacion" / "json" / "rag.json" if path is None else Path(path)
    output_plot = BASE_DIR / "estadisticas" / "graficos" / "frecuencias_rag.png"
    output_tex = BASE_DIR / "estadisticas" / "latex" / "rag_estadisticas.tex"

    if not input_path.exists():
        print(f"Archivo no encontrado: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metricas = {"cobertura": [], "precisión": [], "alucinacion": []}
    chunk_distribuciones = {
        "chunks_rag": [],
        "chunks_reranking": [],
        "chunks_relevantes": [],
        "chunks_omitidos": [],
    }

    for entrada in data:
        evaluacion = entrada.get("evaluacion", {})
        for clave in metricas:
            if clave in evaluacion:
                metricas[clave].append(evaluacion[clave])
        for chunk in chunk_distribuciones:
            if chunk in entrada and isinstance(entrada[chunk], list):
                chunk_distribuciones[chunk].append(len(entrada[chunk]))
            else:
                chunk_distribuciones[chunk].append(0)

    # Preparar LaTeX
    latex = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}

\section*{Análisis RAG}
"""

    # === Moda de categorías ===
    latex += r"\subsection*{Moda de categorías}" + "\n"
    latex += r"\begin{table}[H]" + "\n\\centering\n\\begin{tabular}{ll}\n\\toprule\n"
    latex += r"\textbf{Métrica} & \textbf{Moda (frecuencia)} \\" + "\n\\midrule\n"

    for metrica, valores in metricas.items():
        conteo = Counter(valores)
        moda, freq = conteo.most_common(1)[0]
        print(f"{metrica}: {dict(conteo)} (Moda: {moda})")
        moda_str = str(moda).replace("_", r"\_")
        latex += f"{metrica.capitalize()} & {moda_str} ({freq}) \\\\\n"

    latex += r"\bottomrule\n\end{tabular}\n\caption{Moda por categoría en las métricas RAG}\n\end{table}\n"

    # === Estadísticas de chunks ===
    latex += r"\subsection*{Media y desviación típica de los chunks}" + "\n"
    latex += r"\begin{table}[H]" + "\n\\centering\n\\begin{tabular}{lll}\n\\toprule\n"
    latex += r"\textbf{Campo} & \textbf{Media} & \textbf{Desviación típica} \\" + "\n\\midrule\n"

    for nombre, lista in chunk_distribuciones.items():
        media = statistics.mean(lista)
        std = statistics.stdev(lista) if len(lista) > 1 else 0
        print(f"{nombre}: Media={media:.2f}, Desviación típica={std:.2f}")
        campo_tex = nombre.replace("_", r"\_")
        latex += f"{campo_tex} & {media:.2f} & {std:.2f} \\\\\n"

    latex += r"\bottomrule\n\end{tabular}\n\caption{Media y desviación típica de chunks}\n\end{table}\n"

    # === Gráfico de frecuencias ===
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for metrica, valores in metricas.items():
        conteo = Counter(valores)
        etiquetas = list(conteo.keys())
        cantidades = list(conteo.values())
        plt.bar(etiquetas, cantidades, label=metrica.capitalize(), alpha=0.7)
    plt.title("Frecuencia de categorías RAG")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    # Insertar imagen en LaTeX
    relative_img_path = Path("..") / "graficos" / output_plot.name
    latex += r"""
\subsection*{Gráfico de frecuencias}
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{""" + relative_img_path.as_posix() + r"""}
\caption{Frecuencia de categorías en las métricas RAG}
\end{figure}
"""

    latex += r"\end{document}"

    # Guardar archivo LaTeX
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\nArchivo LaTeX generado en: {output_tex}")
    print(f"Gráfico guardado en: {output_plot}")

if __name__ == "__main__":
    analizar_rag()
