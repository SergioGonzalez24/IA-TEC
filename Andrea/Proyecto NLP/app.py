import spacy
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
from summarizer import Summarizer
import language_tool_python

# Cargar el modelo de idioma
nlp = spacy.load("es_core_news_sm")

# Inicializar LanguageTool para el idioma español
tool = language_tool_python.LanguageToolPublicAPI('es')


def procesar_texto():
    # Obtener el texto de la caja de texto
    oracion = texto_entry.get("1.0",'end-1c')

    # Verificar y corregir ortografía
    matches = tool.check(oracion)
    oracion_corregida = tool.correct(oracion)

    # Mostrar sugerencias de corrección en la interfaz
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, "Sugerencias de Corrección:\n")
    for match in matches:
        output_text.insert(tk.END, f"{match}\n")
    output_text.config(state=tk.DISABLED)

    # Procesar el texto corregido con spaCy
    doc = nlp(oracion_corregida)

    # Obtener información sobre tokens
    df_tokens = pd.DataFrame([{'Texto': token.text, 'Parte de la oración': token.pos_, 'Dependencia': token.dep_} for token in doc])

    # Mostrar información sobre tokens en la interfaz
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, "\n\nInformación sobre Tokens (después de corrección):\n")
    output_text.insert(tk.END, df_tokens.to_string(index=False))
    output_text.config(state=tk.DISABLED)

    # Obtener información sobre entidades nombradas
    df_entidades = pd.DataFrame([(entidad.text, entidad.label_) for entidad in doc.ents], columns=["Entidad", "Tipo"])

    # Mostrar información sobre entidades nombradas en la interfaz
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, "\n\nTabla de Entidades Nombradas:\n")
    output_text.insert(tk.END, df_entidades.to_string(index=False))
    output_text.config(state=tk.DISABLED)

    # Generar un resumen del texto utilizando BERT
    summarizer = Summarizer()
    resumen = summarizer(oracion_corregida, ratio=0.2)

    # Mostrar el resumen en la interfaz
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, "\n\nResumen del Texto:\n")
    output_text.insert(tk.END, resumen)
    output_text.config(state=tk.DISABLED)

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Generador de Resúmenes con spaCy, bert-extractive-summarizer y corrección ortográfica")
root.geometry("800x600")

# Crear una caja de texto para ingresar el texto
texto_label = tk.Label(root, text="Ingrese el texto:")
texto_label.pack()


texto_entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
texto_entry.pack()

# Botón para procesar el texto y generar el resumen
procesar_button = tk.Button(root, text="Generar Resumen", command=procesar_texto)
procesar_button.pack()

# Área de texto para mostrar la salida
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=30)
output_text.pack()
output_text.config(state=tk.DISABLED)

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
