from langchain_core.tools import tool
from compare_codes import compare_codes
from typing import List
from prompts import prompt
from langchain_core.prompts import ChatPromptTemplate

@tool
def compare_directions_by_codes(codes: List[str]) -> dict:
    """
    Сравнивает образовательные коды (профили, направления, УГС) и возвращает:
    - косинусную близость
    - ключевые tf-idf слова
    - темы BERTopic

    Аргументы:
    codes — список кодов (например, ["09.03.01", "09.03.02"])
    """
    template = ChatPromptTemplate.from_template(prompt)
    result = compare_codes(codes=codes)

    prompt_result = template.format_messages(
        codes=", ".join(codes),
        similarities="\n".join([f"{k}: {v}" for k, v in result["similarities"].items()]),
        tfidf="\n".join([
            f"{k}: {', '.join([f'{word} ({round(score, 2)})' for word, score in v])}"
            for k, v in result["tfidf"].items()
        ]),
        topics="\n".join([
            f"{k}: " + "; ".join(
                [", ".join(t['keywords']) for t in v]
            ) for k, v in result["topics"].items()
        ])
    )

    return prompt_result

if __name__ == "__main__":
    print(compare_directions_by_codes(["09.03.02", "15.03.01"]))