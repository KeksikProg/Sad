from langchain_core.tools import tool
from compare_codes import compare_codes
from typing import List
from prompts import prompt
from langchain_core.prompts import ChatPromptTemplate

@tool
def compare_directions_by_codes(codes: List[str]) -> dict:
    """
    Сравнивает образовательные коды ОДНОГО УРОВНЯ:
    - Профили подготовки (формат: xx.xx.xx)
    - Направления подготовки (формат: xx.xx)
    - Укрупнённые группы специальностей (УГС, формат: xx)

    Важно: нельзя смешивать коды разных уровней между собой.
    То есть:
    - Профили сравниваются только с профилями
    - Направления — только с направлениями
    - УГС — только с УГС

    На выходе:
    - Косинусная близость между кодами
    - Ключевые TF-IDF слова для каждого кода
    - Темы BERTopic (топики со словами)

    Аргументы:
    codes — список кодов одного уровня (например, ["09.03.01", "09.03.02"] или ["09.03"] или ["09"])
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