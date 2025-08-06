import pickle
import numpy as np

with open("dataFRIDA.pkl", "rb") as f:
    profile_data = pickle.load(f)

def cosine_sim(vec1, vec2):
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def compare_codes(codes: list[str]) -> str:
    results = {}

    embeddings = []
    tfidf_data = {}
    topic_data = {}

    for code in codes:
        item = profile_data.get(code)
        if not item:
            continue

        tfidf_data[code] = item["tfidf"]
        topic_data[code] = item["topics"]
        embeddings.append(item["embedding"])

    sim_matrix = {}
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            pair = f"{codes[i]} vs {codes[j]}"
            if sim <= 0.93:
                sim_matrix[pair] = "Очень сильно отличаются"
            elif sim > 0.93 and sim <= 0.95:
                sim_matrix[pair] = "Отличаются уклоном обучения, но схожесть есть"
            elif sim > 0.95 and sim <= 0.99:
                sim_matrix[pair] = "Похожи, но с разными акцентами"
            elif sim > 0.99:
                sim_matrix[pair] = "Идентичны"

    results["tfidf"] = tfidf_data
    results["topics"] = topic_data
    results["similarities"] = sim_matrix
    results["codes"] = codes

    return results

#if __name__ == "__main__":
#    print(compare_codes(["15.03.01", "09.03.02", "09.03.01"])["tfidf"])