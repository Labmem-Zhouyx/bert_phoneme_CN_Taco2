""" from https://github.com/keithito/tacotron """
import re
import json
import codecs
from text import cleaners
# from text.symbols import symbols
from .symbols_yinsu import symbols as yinsu_symbols
from .symbols_character import symbols as character_symbols
from .symbols_pinyin import symbols as pinyin_symbols

from .pinyin import split_pinyin
# from .symbols import symbols
# from .symbols_yinsu import symbols


# Mappings from symbol to numeric ID and vice versa:
__yinsu_symbol_to_id = {s: i for i, s in enumerate(yinsu_symbols)}
_id_to_yinsu_symbol = {i: s for i, s in enumerate(yinsu_symbols)}

__character_symbol_to_id = {s: i for i, s in enumerate(character_symbols)}
_id_to_character_symbol = {i: s for i, s in enumerate(character_symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def pinyin_to_yinsu(class2idx_file):
  with codecs.open(class2idx_file, 'r', 'utf-8') as usernames:
    class2idx = json.load(usernames)
  pinyin_to_yinsu_dict = []
  for k,v in class2idx.items():
    # print('CHECK K', k)
  # for pinyin in class2idx:
    if k == 'r5':
      k = 'er5'
    if k == 'y2':
      k = 'yi2'
    if k == 'q':
      k = 'qu'
    if len(k.split(' ')) > 1:
      # print('CHECK Continue:', k)
      pinyin_to_yinsu_dict.append([0, 0, 0, 0])
      continue
    yinsus = list(split_pinyin(k))
    # print('CHECK yinsus', yinsus)
    if yinsus[0] == '':
      yinsus[0] = 'nan'
    if yinsus[1] == '':
      yinsus[1] = 'nan'
    if yinsus[2] == '':
      yinsus[2] = 'nan'
    if yinsus[3] == '':
      yinsus[3] = 'nan'
    yinsu_id = []
    for yinsu in yinsus:
      yinsu_id.append(__yinsu_symbol_to_id[yinsu])
    pinyin_to_yinsu_dict.append(yinsu_id)
  return pinyin_to_yinsu_dict



def text_to_sequence(text, cleaner_names):
  sequence = []

  words = text.split(' ')
  for word in words:
    # print('CHECK word:', word)
    yinsus = split_pinyin(word)
    # print('CHECK split_pinyin:', yinsus)
    yinsu_id = []
    for yinsu in yinsus:
      if yinsu != '':
        yinsu_id.append(__yinsu_symbol_to_id[yinsu])
    # print('CHECK yinsu_id:', yinsu_id)
    sequence.extend(yinsu_id)

  return sequence


def poly_yinsu_to_sequence(text, original_poly_yinsu, polyphone_dict):
  words = []
  words_id = []
  poly_yinsu_sequence = []
  select_target_sequence = []

  sequence_to_check = []
  sequence_to_check_id = []

  for word in text:
    words.append(word)
    words_id.append(__character_symbol_to_id[word])
  original_poly_yinsu = original_poly_yinsu.split(' ')
  # print('CHECK original_poly_yinsu', original_poly_yinsu)

  for char, original_pinyin in zip(words, original_poly_yinsu):
    poly_list = polyphone_dict[char]
    # 24-D (phoneme, id) for every character
    poly_yinsus = []
    poly_yinsus_id = []
    # 6-D list of select target (float) for every character
    select_target = []
    for pinyin in poly_list:
      if pinyin == 'xxx':
        poly_yinsus.extend(['xx', 'xx', 'xx', 'xx'])
      else:
        yinsus = list(split_pinyin(pinyin))
        if yinsus[0] == '':
          yinsus[0] = 'ns'
        if yinsus[1] == '':
          yinsus[1] = 'ny'
        if yinsus[2] == '':
          yinsus[2] = 'nrr'
        assert yinsus[3] != ''
        # poly_yinsus.extend(yinsus)
        poly_yinsus.extend(yinsus)
      if original_pinyin == pinyin:
        select_target.append(1.0)
      else:
        select_target.append(0.0)
    for poly_yinsu in poly_yinsus:
      poly_yinsus_id.append(__yinsu_symbol_to_id[poly_yinsu])

    # poly_yinsu_sequence.append(poly_yinsus_id)
    # select_target_sequence.append(select_target)
    poly_yinsu_sequence.extend(poly_yinsus_id)
    select_target_sequence.extend(select_target)

    # target_to_check = int(select_target.index(max(select_target)))
    # yinsu_to_check = []
    # yinsu_to_check_id = []
    # for i in [0, 1, 2, 3]:
      # yinsu_to_check.append(poly_yinsus[target_to_check*4 + i])
      # yinsu_to_check_id.append(poly_yinsus_id[target_to_check*4 + i])
    # sequence_to_check.extend(yinsu_to_check)
    # sequence_to_check_id.extend(yinsu_to_check_id)

  return words, words_id, poly_yinsu_sequence, select_target_sequence
  # print('CHECK yinsu_to_check', sequence_to_check)
  # return sequence_to_check_id


er_phoneme = ['er1', 'er2', 'er3', 'er4', 'er5', 'rr1', 'rr2', 'rr3', 'rr4', 'rr5']


def poly_yinsu_to_mask(text, original_poly_yinsu, mask_dict):
  words = []
  words_id = []
  mask_sequence = []
  target_sequence = []

  sequence_to_check = []
  sequence_to_check_id = []

  for word in text:
    words.append(word)
    words_id.append(__character_symbol_to_id[word])
  original_poly_yinsu = original_poly_yinsu.split(' ')
  # print('CHECK original_poly_yinsu', original_poly_yinsu)

  # 1-D list of select target (float) for every character
  target_sequence = []
  for char, original_pinyin in zip(words, original_poly_yinsu):
    poly_pinyin_list = mask_dict[char]
    # Not fixed mask (to make 1539 mask in model) for every character
    mask_list = []
    for (pinyin, id) in  poly_pinyin_list.items():
      mask_list.append(id)

      if pinyin in er_phoneme:
        pinyin = pinyin
      else:
        if pinyin[-2] == 'r':
          one_v = pinyin[:-2] + pinyin[-1]
        else:
          pinyin = pinyin

      if original_pinyin == pinyin:
        target_sequence.append(id)
    mask_sequence.append(mask_list)

  # print('CHECK words', words)
  # print('CHECK original_poly_yinsu', original_poly_yinsu)
  # print('CHECK mask_sequence', mask_sequence)
  # print('CHECK target_sequence', target_sequence)

  return words, words_id, mask_sequence, target_sequence
  
  
def poly_yinsu_to_mask_inference(text, mask_dict):
  words = []
  words_id = []
  mask_sequence = []

  for word in text:
    words.append(word)
    words_id.append(__character_symbol_to_id[word])

  for char in words:
    poly_pinyin_list = mask_dict[char]
    # Not fixed mask (to make 1539 mask in model) for every character
    mask_list = []
    for (pinyin, id) in  poly_pinyin_list.items():
      mask_list.append(id)
    mask_sequence.append(mask_list)

  # return words, words_id, mask_sequence
  return words, mask_sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _yinsus_to_sequence(symbols):
  return [[_id_to_yinsu_symbol[ss] for ss in s] for s in symbols]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s != '_' and s != '~'
