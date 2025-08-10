from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

with open("ABS_FULL.pkl", "rb") as f:
    profile_data = pickle.load(f)

def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

def compare_codes(codes: List[str]) -> Dict:
    """
    Сравнивает несколько направлений по кодам.
    Возвращает embeddings-схожесть, TF-IDF, темы и краткие описания (summary).
    """
    results = {
        "codes": codes,
        "tfidf": {},
        "topics": {},
        "summaries": {},  # ← Новое поле
        "similarities": {}
    }

    # Список для embeddings (в порядке codes)
    valid_codes = []
    embeddings = []

    for code in codes:
        item = profile_data.get(code)
        if not item:
            continue  # Пропускаем несуществующие коды

        # Собираем данные
        results["tfidf"][code] = item["tfidf"]
        results["topics"][code] = item["topics"]
        results["summaries"][code] = item.get("summary", "").strip() or "Описание отсутствует."  # если пусто — подсказка

        # Только если есть embedding, добавляем в сравнение
        if "embedding" in item and item["embedding"] is not None:
            embeddings.append(item["embedding"])
            valid_codes.append(code)
        else:
            # Если нет embedding, считаем, что не можем сравнивать
            results["similarities"][f"{code} vs ?"] = "Нет данных для сравнения (embedding)"

    # Сравнение по cosine similarity между embedding
    sim_matrix = {}
    for i in range(len(valid_codes)):
        for j in range(i + 1, len(valid_codes)):
            code1, code2 = valid_codes[i], valid_codes[j]
            sim = cosine_sim(embeddings[i], embeddings[j])

            pair = f"{code1} vs {code2}"
            if sim <= 0.93:
                sim_matrix[pair] = "Очень сильно отличаются"
            elif sim <= 0.95:
                sim_matrix[pair] = "Отличаются"
            elif sim <= 0.99:
                sim_matrix[pair] = "Похожи, но с разными акцентами"
            else:  # sim > 0.99
                sim_matrix[pair] = "Идентичны"

    results["similarities"].update(sim_matrix)

    return results

#if __name__ == "__main__":
#    print(compare_codes(["15.03.01", "09.03.02", "09.03.01"])["summaries"])