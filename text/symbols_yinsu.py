'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

# Mandarin initials (普通话声母列表)  21
_initials = ['b', 'p', 'f', 'm', \
             'd', 't', 'n', 'l', \
             'g', 'k', 'h', \
             'j', 'q', 'x', \
             'zh', 'ch', 'sh', 'r', \
             'z', 'c', 's']

# Mandarin finals (普通话韵母列表) 65
_finals = ['a',  'ai', 'ao',  'an',  'ang', \
           'o',  'ou', 'ong', \
           'e',  'ei', 'en',  'eng', 'er', 'ev', \
           'i',  'ix', 'iy', \
           'ia', 'iao','ian', 'iang','ie', \
           'in', 'ing','io',  'iou', 'iong', \
           'u',  'ua', 'uo',  'uai', 'uei', \
           'uan','uen','uang','ueng', \
           'v',  've', 'van', 'vn', \
           'ng', 'mm', 'nn']

# Retroflex (Erhua) (儿化音信息) 64
_retroflex = ['rr']

# Tones (声调信息)
_tones = ['1', '2', '3', '4', '5']



_nan = ['nan']

# # Prosodic structure symbols (韵律结构标记)
# _prosodic_struct = [' ', '/', ',', '.', '?', '!']
_eos = ['~']

_pad = ['_']

symbols = _pad + _initials + _finals + _tones + _nan + _eos + _retroflex
