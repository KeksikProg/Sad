import re

def get_topic_distributions(df, topic_model, top_n=3):
    result = {}

    for code in df['code'].unique():
        subset = df[df['code'] == code]
        topic_counts = Counter(subset['topic'])
        total = sum(topic_counts.values())

        # Получаем top_n тем с их долей
        top_topics = topic_counts.most_common(top_n)
        topic_summary = []

        for topic_id, count in top_topics:
            weight = round((count / total) * 100, 1)  # процент
            words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]  # топ-5 слов
            topic_summary.append({
                "topic_id": topic_id,
                "weight_percent": weight,
                "keywords": words
            })

        result[code] = topic_summary

    return result

def extract_cleaned_text(text):
    """
    Извлекает из аннотации только релевантную информацию:
    - Название дисциплины
    - Цель изучения дисциплины
    - Задачи изучения дисциплины
    - Компетенции (если есть)
    """
    if not isinstance(text, str):
        return ""

    # 1. Название дисциплины
    title_match = re.search(r'АННОТАЦИЯ\s+к\s+рабочей\s+программе\s+дисциплины\s+[«"]([^»"]+)[»"]', text, re.IGNORECASE)
    if title_match == None:
      title_match = re.search(r'АННОТАЦИЯ\s+к\s+рабочей\s+программе\s+практики\s+[«"]([^»"]+)[»"]', text, re.IGNORECASE)
    title = f"{title_match.group(1)}" if title_match else ""

    # 2. Цель дисциплины
    goal_match = re.search(r'(Цел(?:ь|и)?(?:\s+изучения)?\s+дисциплины[:\s]*)(.*?)(?:Задачи изучения дисциплины:|Перечень формируемых компетенций:|Общая трудоемкость дисциплины:|Форма итогового контроля|$)',
      text,
      re.DOTALL | re.IGNORECASE)
    goal = goal_match.group(2).strip() if goal_match else ""

    # 3. Задачи дисциплины
    task_match = re.search(
        r'Задачи\s+изучения\s+дисциплины:(.*?)(Перечень формируемых компетенций:|Общая трудоемкость дисциплины:|Форма итогового контроля|$)',
        text, re.DOTALL | re.IGNORECASE)
    if task_match == None:
      task_match = re.search(
        r'Задачи\s+изучения\s+практики:(.*?)(Перечень формируемых компетенций:|Общая трудоемкость дисциплины:|Форма итогового контроля|$)',
        text, re.DOTALL | re.IGNORECASE)
    tasks = task_match.group(1).strip() if task_match else ""

    # 4. Компетенции (если есть)
    comp_match = re.search(
        r'Перечень\s+формируемых\s+компетенций:(.*?)(Общая трудоемкость дисциплины:|Форма итогового контроля|$)',
        text, re.DOTALL | re.IGNORECASE)
    competences = comp_match.group(1).strip() if comp_match else ""

    # Сборка итогового текста
    cleaned_parts = []
    if title:
        cleaned_parts.append(f"{title}")
    if goal:
        cleaned_parts.append(f"\nЦель:\n{goal}")
    if tasks:
        cleaned_parts.append(f"\nЗадачи:\n{tasks}")
    if competences:
        cleaned_parts.append(f"\nКомпетенции:\n{competences}")

    return "\n".join(cleaned_parts).strip()

def get_mean_embedding(df, code):
    embs = torch.stack(df[df['code'] == code]['embedding'].tolist())
    return embs.mean(dim=0)

def parse_code(code):
    parts = code.split(".")
    return {
        "group_code": parts[0],
        "direction_code": ".".join(parts[:2]),
        "profile_code": ".".join(parts)
    }

