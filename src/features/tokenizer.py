from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE, WordPiece
from tokenizers.pre_tokenizers import Split, Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

CHARS: list = [
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

quartznet_vocab = {"[BLANK]": 0, "[UNK]": 1, " ": 2}
for i, char in enumerate(CHARS):
    quartznet_vocab[char] = i + 3

QUARTZNET_TOKENIZER = Tokenizer(WordLevel(vocab=quartznet_vocab, unk_token="[UNK]"))
QUARTZNET_TOKENIZER.pre_tokenizer = Split(pattern=r"", behavior="isolated") 

CITRINET_TOKENIZER = Tokenizer(BPE(unk_token="[UNK]"))
CITRINET_TOKENIZER.pre_tokenizer = Metaspace(replacement='_')
CITRINET_TOKENIZER.decoder = MetaspaceDecoder(replacement='_')
CITRINET_TRAINER = BpeTrainer(vocab_size=256, initial_alphabet=CHARS, special_tokens=["[BLANK]", "[UNK]"])

CONFORMER_TOKENIZER = Tokenizer(WordPiece(unk_token="[UNK]"))
CONFORMER_TOKENIZER.pre_tokenizer = Metaspace(replacement='_')
CONFORMER_TOKENIZER.decoder = MetaspaceDecoder(replacement='_')
CONFORMER_TRAINER = WordPieceTrainer(vocab_size=1024, initial_alphabet=CHARS, special_tokens=["[BLANK]", "[UNK]"])

FASTCONFORMER_TOKENIZER = Tokenizer(WordPiece(unk_token="[UNK]"))
FASTCONFORMER_TOKENIZER.pre_tokenizer = Metaspace(replacement='_')
FASTCONFORMER_TOKENIZER.decoder = MetaspaceDecoder(replacement='_')
FASTCONFORMER_TRAINER = WordPieceTrainer(vocab_size=1024, initial_alphabet=CHARS, special_tokens=["[BLANK]", "[UNK]"])
