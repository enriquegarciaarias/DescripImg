from sources.common.common import logger, processControl, log_

from sentence_transformers import SentenceTransformer, util
import torch
import spacy
import re

nlp = spacy.load("en_core_web_sm")  # Para segmentar párrafos con más precisión

MODEL_NAME_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME_EMBEDDING)
embedding_model.to("cuda" if torch.cuda.is_available() else "cpu")


def extract_entity(text):
    """
    Extrae entidades de dos maneras:
    1. Si el texto termina en "de X", donde X es una sola palabra con mayúscula inicial, extrae "X".
    2. Extrae todas las palabras completamente en mayúsculas dentro del texto, excepto la primera.
    """
    # Extraer entidad en el formato "de X"
    match = re.search(r'de ([A-ZÁÉÍÓÚÑ][a-záéíóúñ]*)$', text)
    entity = [match.group(1)] if match else []

    # Extraer todas las palabras completamente en mayúsculas
    words_in_caps = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{2,}\b', text)  # Captura palabras de 2+ letras en mayúsculas
    if words_in_caps:
        words_in_caps = words_in_caps[1:]  # Excluye la primera palabra si hay varias en mayúsculas

    return list(dict.fromkeys(entity + words_in_caps))  # Elimina duplicados

def extract_subject(text):
    doc = nlp(text)
    keywords = []

    # Lista de palabras irrelevantes al inicio
    ignore_words = {"reconstrucción", "planta", "el", "la", "del", "de", "desde"}

    # Identificar sustantivos y nombres propios
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and token.text.lower() not in ignore_words:
            keywords.append(token.text)

    return keywords

def convert_docx_to_txt(input_path, output_path=None):
    from docx import Document
    """
    Convierte un archivo DOCX a TXT extrayendo todo el texto.

    :param input_path: Ruta del archivo .docx de entrada.
    :param output_path: Ruta del archivo .txt de salida.
    """
    doc = Document(input_path)
    text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    if output_path:
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
    return text


def buildContextData(documentContextpath, title, top_n=3, threshold=0.5):
    """
    Extrae los párrafos más relevantes basados en palabras clave y similitud semántica.
    """
    entity = extract_entity(title)  # Metodo basado en regex
    subject = extract_subject(title)  # Ahora devuelve una lista

    # Asegurar que entity es una lista (en caso de que sea una cadena)
    if isinstance(entity, str):
        entity = [entity]

    # Unir title, entity y subject en una única lista sin duplicados
    keywords = list(dict.fromkeys([title] + (entity if entity else []) + (subject if subject else [])))

    text = convert_docx_to_txt(documentContextpath)
    paragraphs = [para.text for para in nlp(text).sents]  # Segmentación precisa

    # Obtener embeddings de keywords y párrafos
    keyword_embeddings = embedding_model.encode(keywords, convert_to_tensor=True)
    paragraph_embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True)

    # Promedio de embeddings de keywords
    keyword_embedding = torch.max(keyword_embeddings, dim=0, keepdim=True)[0]

    # Calcular similaridad coseno
    similarities = util.pytorch_cos_sim(keyword_embedding, paragraph_embeddings)[0]

    # Filtrar por umbral de similitud
    top_indices = similarities.argsort(descending=True)
    filtered_indices = [i for i in top_indices if similarities[i] >= threshold]
    a = similarities.min().item()
    b = similarities.max().item()
    if len(filtered_indices) == 0 and similarities.max().item() >= 0.3:
        min_threshold = similarities.max().item() * 0.8  # Baja el umbral al 80% del valor más alto
        filtered_indices = [i for i in top_indices if similarities[i] >= min_threshold]

    # Seleccionar los top_n párrafos con similitud suficiente
    relevant_paragraphs = [paragraphs[i] for i in filtered_indices[:top_n]]

    # Construir contexto final
    contexto = ". ".join(relevant_paragraphs)
    contexto = re.sub(r'\n+', ' ', contexto).strip()
    contexto = re.sub(r'\b[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ]+\b', ' ', contexto)  # Reemplaza con espacio en lugar de eliminar
    contexto = re.sub(r'\s+', ' ', contexto).strip()  # Normaliza los espacios

    # Verificar si contiene la entidad o el sujeto clave
    check_words = (entity if isinstance(entity, list) else []) + (subject if isinstance(subject, list) else [])

    # Check if at least one word is in contexto
    if check_words and not any(word in contexto for word in check_words):
        return None, keywords
    return contexto, keywords
