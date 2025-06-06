from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.pre_tokenizers import Split, Whitespace
from tokenizers.trainers import WordPieceTrainer


CHARS: list = [
    ' ',
    'а',
    'б',
    'в',
    'г',
    'д',
    'е',
    'ж',
    'з',
    'и',
    'й',
    'к',
    'л',
    'м',
    'н',
    'о',
    'п',
    'р',
    'с',
    'т',
    'у',
    'ф',
    'х',
    'ц',
    'ч',
    'ш',
    'щ',
    'ъ',
    'ы',
    'ь',
    'э',
    'ю',
    'я',
]
INT_TO_CHAR = dict(enumerate(CHARS))
CHAR_TO_INT = {v: k for k, v in INT_TO_CHAR.items()}


QUARTZNET_TOKENIZER = Tokenizer(WordLevel(vocab=CHAR_TO_INT, unk_token=" "))
QUARTZNET_TOKENIZER.pre_tokenizer = Split(r"", "isolated")

CITRINET_TOKENIZER = Tokenizer(WordPiece())
CITRINET_TOKENIZER.pre_tokenizer = Whitespace()
CITRINET_TRAINER = WordPieceTrainer(vocab_size=256, initial_alphabet=CHARS, special_tokens=["[PAD]"])
