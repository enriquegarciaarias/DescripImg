from _ast import comprehension

from sources.common.common import logger, processControl, log_

from sentence_transformers import SentenceTransformer, util
from unidecode import unidecode
import torch
import spacy
import re

#nlp = spacy.load("en_core_web_sm")  # Para segmentar párrafos con más precisión Inglés
nlp = spacy.load("es_core_news_lg") # Para segmentar párrafos con más precisión Castellano

MODEL_NAME_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME_EMBEDDING)
embedding_model.to("cuda" if torch.cuda.is_available() else "cpu")


def extract_entity(text):
    """
    Extrae entidades de dos maneras:
    1. Si el texto contiene un patrón tipo "[palabra] de [Entidad]", como "templo de Némesis", extrae ambas: "Némesis" y "templo de Némesis".
    2. Extrae todas las palabras completamente en mayúsculas dentro del texto, excepto la primera.
    """
    entity = []

    # Extraer "X de Y" donde Y empieza con mayúscula
    match = re.search(r'([\wáéíóúñÁÉÍÓÚÑ]+)\s+de\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)', text)
    if match:
        base = match.group(1)
        nombre = match.group(2)
        entity.append(nombre)  # "Némesis"
        entity.append(f"{base} de {nombre}")  # "templo de Némesis"

    # Extraer palabras completamente en mayúsculas (excepto la primera)
    words_in_caps = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{2,}\b', text)
    if words_in_caps:
        words_in_caps = words_in_caps[1:]

    # Unificar y eliminar duplicados manteniendo orden
    return list(dict.fromkeys(entity + words_in_caps))


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

def buildKeywords(title):
    entity = extract_entity(title)  # Metodo basado en regex
    subject = extract_subject(title)  # Ahora devuelve una lista

    # Asegurar que entity es una lista (en caso de que sea una cadena)
    if isinstance(entity, str):
        entity = [entity]

    # Unir title, entity y subject en una única lista sin duplicados
    keywords = list(dict.fromkeys([title] + (entity if entity else []) + (subject if subject else [])))
    cleanList = [unidecode(word) for word in keywords]
    return cleanList


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
    return re.sub(r'\n{2,}', ' ', text)


def asignarToposSpacy(bloques, topos2):
    for bloque in bloques:
        texto_sin_acentos = unidecode(bloque['text']).lower()  # Normaliza y pasa a minúsculas
        bloque['topoSpacy'] = [
            topo for topo in topos2
            if unidecode(topo).lower() in texto_sin_acentos
        ]
    return bloques


def indexacionBloquesFromText(text):
    """
    Construye una lista con los párrafos asignados a los diferentes topónimos
    """

    prefijos = ["recinto funerario", "santuario", "templo", "fortaleza", "yacimiento", "estatua", "templete", "recinto"]
    topos1 = toponimos(text, prefijos)
    topos2 = toponimosSpacy(text)
    # bloques = bloquesTextoToponimos(text, topos2)
    paragraphs = [sent.text.strip() for sent in nlp(text).sents if len(sent.text.strip().split()) >= 8]
    paragraphs = completar_parrafos_con_toponimos(paragraphs, prefijos)
    bloques = asignar_parrafos_a_toponimos(paragraphs, prefijos)
    bloques = asignarToposSpacy(bloques, topos2)
    return bloques

def buildContextData(documentContextpath, title, top_n=3, threshold=0.5):
    text = convert_docx_to_txt(documentContextpath)
    keywords = buildKeywords(title)
    bloques = indexacionBloquesFromText(text)

    contexto = {}

    # Identifica y almacena bloques que contengan keywords
    textosUsados = []
    for bloque in bloques:
        if bloque['text'] not in textosUsados:
            for elemento in keywords:
                peso = len(elemento.split())

                if elemento in bloque.get('toponimos', []):
                    contexto.setdefault(peso, []).append(bloque['text'])
                    textosUsados.append(bloque['text'])
                elif elemento in bloque.get('topoSpacy', []):
                    contexto.setdefault(peso, []).append(bloque['text'])
                    textosUsados.append(bloque['text'])

    #Si no hay resultado pero si hay keywords ? por ejemplo "el gimnasio"

    resultado = []
    contador = 0

    # Ordenar claves numéricamente en orden descendente
    for clave in sorted(contexto.keys(), key=int, reverse=True):
        for texto in contexto[clave]:
            if contador < top_n:
                resultado.append(texto)
                contador += 1
            else:
                break
        if contador >= top_n:
            break

    # Unir con ".  " como separador
    texto_final = ".  ".join(resultado)
    log_("info", logger, f"texto contexto: {texto_final}, keywords: {keywords}")

    return texto_final, keywords

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


def toponimos(text, prefijos):
    text = re.sub(r'\n+', ' ', text)

    # Prefijos relevantes


    # Unimos los prefijos en una expresión alternada
    prefijos_regex = '|'.join([re.escape(p) for p in prefijos])

    # Patrón general que captura todo el bloque después del prefijo hasta el siguiente punto o coma
    pattern = rf"\b(?:{prefijos_regex})\s+(?:[a-zA-ZáéíóúÁÉÍÓÚñÑ]+\s+)*[A-ZÁÉÍÓÚÑ][a-záéíóúñÁÉÍÓÚÑ]+"

    # Buscar coincidencias
    matches = re.findall(pattern, text)

    # Limpieza: quitar duplicados y espacios innecesarios
    toponimos = sorted(set([m.strip() for m in matches]))
    a = list(toponimos)
    cleanList = [unidecode(word) for word in a]
    return cleanList

def toponimosSpacy(text):
    # Cargar el modelo de spaCy (modelo en español)
    nlp = spacy.load("es_core_news_lg")
    doc = nlp(text)

    # Extraer entidades etiquetadas por spaCy
    entidades_lugar = [
        ent.text.strip() for ent in doc.ents
        if ent.label_ in ["GPE", "LOC", "PER"]
    ]

    # Heurística adicional: incluir nombres propios (PROPN) que no estén al inicio de frase
    nombres_propios = [
        token.text.strip() for token in doc
        if token.pos_ == "PROPN"
        and token.text[0].isupper()
        and not token.is_sent_start
        and len(token.text) > 3
    ]

    # Combinar listas y eliminar duplicados
    toponimos = sorted(set(entidades_lugar + nombres_propios))

    #log_("info", logger, f"Lista de topónimos extraídos: {toponimos}")
    cleanList = [unidecode(word) for word in toponimos]
    return cleanList


def bloquesTextoToponimos(paragraphs, toponimos):
    """
    Esta segmentacion se basa en saltos de linea y pueden haber saltos en un ismo parrafo, asi que eliminé los saltos antes y uso spacy de otra forma
    doc = nlp(text)
    paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
    """


    # Estructura para guardar asociaciones
    bloques_toponimicos = []

    # Asociar cada párrafo con uno o más topónimos
    for i, para in enumerate(paragraphs):
        encontrados = [top for top in toponimos if re.search(rf'\b{re.escape(top)}\b', para)]
        if encontrados:
            bloques_toponimicos.append({
                "id": f"parrafo_{i + 1}",
                "toponimos": encontrados,
                "text": para
            })

    return bloques_toponimicos

def completar_parrafos_con_toponimos(paragraphs, prefijos_base):
    """
    Recorre los párrafos e inserta el topónimo completo cuando
    solo se menciona el prefijo (e.g., "el santuario").
    """
    # Preparar regex por cada prefijo base
    prefijos_info = {}
    for base in prefijos_base:
        base_esc = re.escape(base)
        prefijos_info[base] = {
            "singular": re.compile(rf"\b{base_esc}\s+de\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"),
            "plural": re.compile(rf"\b{base_esc}s\s+de\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+y\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?"),
            "mencion_simple": re.compile(rf"\b{base_esc}s?\b")
        }

    # Último topónimo conocido por prefijo
    ultimo_toponimo = {base: None for base in prefijos_base}

    nuevos_parrafos = []

    for para in paragraphs:
        nuevo_para = para  # base modificable

        for base, regex in prefijos_info.items():
            # ¿Hay topónimo completo en este párrafo?
            if regex["plural"].search(para):
                match = regex["plural"].search(para)
                ultimo_toponimo[base] = match.group(0)
            elif regex["singular"].search(para):
                match = regex["singular"].search(para)
                ultimo_toponimo[base] = match.group(0)
            # ¿Hay solo mención del prefijo y ya tenemos uno anterior?
            elif regex["mencion_simple"].search(para) and ultimo_toponimo[base]:
                def reemplazo(m):
                    return ultimo_toponimo[base]
                nuevo_para = regex["mencion_simple"].sub(reemplazo, nuevo_para)

        nuevos_parrafos.append(nuevo_para)

    return nuevos_parrafos


def asignar_parrafos_a_toponimos(paragraphs, prefijos_base):
    # 1. Preparar regex por cada prefijo:
    prefijos_info = {}
    for base in prefijos_base:
        base_pat = re.escape(base)
        singular = rf"(?P<prefix>{base_pat})\s+de\s+(?P<one>[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)"
        plural   = rf"(?P<prefix>{base_pat}s)\s+de\s+(?P<first>[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)(?:\s+y\s+(?P<second>[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+))?"
        prefijos_info[base] = {
            'singular': re.compile(singular),
            'plural':   re.compile(plural)
        }

    bloques = []
    toponimo_actual = None
    prefijo_actual = None

    for idx, para in enumerate(paragraphs):
        encontrados = []

        # 2. Buscar todos los patrones en el párrafo
        for base, pats in prefijos_info.items():
            # Buscar todas las coincidencias plurales
            for m in pats['plural'].finditer(para):
                first = f"{base} de {m.group('first')}"
                encontrados.append(first)
                if m.group('second'):
                    second = f"{base} de {m.group('second')}"
                    encontrados.append(second)

            # Buscar todas las coincidencias singulares
            for m2 in pats['singular'].finditer(para):
                encontrados.append(f"{base} de {m2.group('one')}")

        # 3. Si encontramos topónimos completos, actualizamos estado
        if encontrados:
            toponimo_actual = encontrados[0]   # usamos el primero como referencia
            prefijo_actual = next((b for b in prefijos_base if toponimo_actual.startswith(b)), None)
            asignados = encontrados

        else:
            # 4. Si no, revisamos si el prefijo (sin "de...") aparece para heredar
            hereda = False
            for base, pats in prefijos_info.items():
                if re.search(rf"\b{base}s?\b", para):  # singular o plural del prefijo
                    if base == prefijo_actual:
                        hereda = True
                    break
            asignados = [toponimo_actual] if hereda and toponimo_actual else []

        # 5. Guardar bloque si hay topónimos asignados
        if asignados:
            bloques.append({
                "id": f"parrafo_{idx+1}",
                "toponimos": asignados,
                "text": para
            })

    return bloques