# utils_viz.py
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def project_2d(embs: np.ndarray) -> np.ndarray:
    n = embs.shape[0]
    if n >= 3:
        D = cosine_distances(embs)
        return MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(D)
    elif n == 2:
        return PCA(n_components=2).fit_transform(embs)
    else:
        return np.hstack([embs, np.zeros((n,1))])[:, :2]

def plot_positions_2d(coords: np.ndarray, labels: list[str]) -> bytes:
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(coords[:,0], coords[:,1])
    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (coords[i,0], coords[i,1]))
    plt.title("Сравнение (2D проекция)")
    plt.tight_layout()
    return _to_png(fig)

def plot_tfidf_bars(tfidf_items: list[tuple[str,float]], code: str, top_k: int = 10) -> bytes:
    words = [w for w, _ in tfidf_items[:top_k]]
    scores = [float(s) for _, s in tfidf_items[:top_k]]
    y = np.arange(len(words))
    fig = plt.figure(figsize=(7, 4))
    plt.barh(y, scores)
    plt.yticks(y, words)
    plt.gca().invert_yaxis()
    plt.title(f"Топ‑слова — {code}")
    plt.tight_layout()
    return _to_png(fig)

def plot_topics_text(topics: list[dict], code: str, top_topics: int = 3, top_words: int = 6) -> bytes:
    blocks = []
    for t in topics[:top_topics]:
        kws = ", ".join(t.get("keywords", [])[:top_words])
        blocks.append(f"• {kws}")
    text = "\n".join(blocks) if blocks else "нет тем"
    fig = plt.figure(figsize=(6, 3))
    plt.axis("off")
    plt.title(f"Основные темы — {code}", pad=10)
    plt.text(0.02, 0.98, text, va="top", ha="left", wrap=True)
    plt.tight_layout()
    return _to_png(fig)

def plot_similarity_heatmap(codes: list[str], embs: np.ndarray) -> bytes:
    import seaborn as sns  # если не хочешь seaborn — убери, сделай plt.imshow
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embs)
    fig = plt.figure(figsize=(5, 4))
    try:
        sns.heatmap(sim, annot=True, fmt=".2f", xticklabels=codes, yticklabels=codes)
    except Exception:
        plt.imshow(sim)
        plt.xticks(range(len(codes)), codes, rotation=45, ha="right")
        plt.yticks(range(len(codes)), codes)
    plt.title("Семантическое сходство")
    plt.tight_layout()
    return _to_png(fig)
