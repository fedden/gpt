from typing import Callable, List
from random import randbytes, seed


def hex_colour_for_token(token_string: str) -> str:
    seed(hash(token_string))
    return "#" + randbytes(3).hex().upper()
    

def text_tag(text: str, text_colour: str, background_colour: str) -> str:
    return f"<text style='padding: 2px; font-family: monospace; color:{text_colour}; background-color:{background_colour}'>{text}</text>"


def to_html_text(token_strings: List[str]):
    html_string = ""
    for token_string in token_strings:
        for i, sub_string in enumerate(token_string.split("\n")):
            html_string += text_tag(
                text=sub_string.replace(" ", "&nbsp;"), 
                text_colour="white",
                background_colour=hex_colour_for_token(token_string),
            )
            if i > 0:
                html_string += "<br/>"
    return html_string


def render_corpus_as_token_html(corpus: str, token_merge_history: List["TokenPair"], vocab: "Vocab", tokenise: Callable):
    token_strings, token_ids = tokenise(corpus, token_merge_history, vocab)
    corpus_html: str = to_html_text(token_strings)
    if token_merge_history:
        header: str = f"<h1>Iteration {len(token_merge_history)}, merged '{token_merge_history[-1].token_a}' and '{token_merge_history[-1].token_b}', vocab size: {len(vocab)}</h1>"
    else:
        header = f"<h1>Iteration 0, initial character-level tokens, vocab size: {len(vocab)}</h1>"
    return header + "<br/>" + corpus_html 