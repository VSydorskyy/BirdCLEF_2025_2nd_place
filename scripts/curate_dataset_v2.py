import argparse
import os
from glob import glob

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from code_base.utils.main_utils import get_device, write_json

TRAIN_AUDIO_FILTERS = {  # list of tuple (start, end) and/or float (time of vocalization)
    "1139490/CSA36385": ["m", (0.9, 7.3)],
    "1139490/CSA36389": ["m", (0.8, 8.3)],
    "1192948/CSA36358": ["m", (0.8, 7.7)],
    "1192948/CSA36366": ["m", (0.9, 7.7)],
    "1192948/CSA36373": ["m", (0.9, 7.8)],
    "1192948/CSA36388": ["m", (0.8, 7.9)],
    "1194042/CSA18783": ["m", (0.9, 18.5), (20, 26)],
    "1194042/CSA18794": ["m", (0, 8.546)],
    "1194042/CSA18802": ["m", (0.9, 14.2), (25, 30)],
    "126247/XC941297": ["m", (0, None)],
    "126247/iNat1109254": ["m", (0, None)],
    "126247/iNat146584": ["m", (0, None)],
    "126247/iNat320679": ["m", (0, None)],
    "126247/iNat888527": ["m", (0, None)],
    "126247/iNat888729": ["m", (0, None)],
    "1346504/CSA18784": ["m", (0.8, 38.2)],
    "1346504/CSA18791": ["m", (0, 71.906)],
    "1346504/CSA18792": ["m", (0, 21)],
    "1346504/CSA18793": ["m", (0.9, 27.6)],
    "1346504/CSA18803": ["m", (0.9, 93.7)],
    "134933/XC941298": ["m", (0, None)],
    "134933/iNat1108984": ["m", (1, 6), (11, 16), (21, 27)],
    "134933/iNat1160199": ["m", (0, 20), (29, 47)],
    "134933/iNat859408": ["m", (0, None)],
    "135045/iNat1122209": ["m", (0, 10), (12, 22), (24, 32), (36, 46)],
    "135045/iNat1207345": ["m", (9, 19), (34, 44), (64, 73), (69, 78), (78, 108), (105, 120), (120, 128)],
    "135045/iNat1207347": [
        "m",
        (4, 14),
        (17, 42),
        (42, 50),
        (49.5, 58),
        (57, 66),
        (67, 76),
        (77, 87),
        (90, 100),
        (104, 111.4),
    ],
    "135045/iNat1208549": [
        "m",
        (9, 19),
        (27, 36),
        (51, 60.5),
        (64, 73),
        (78, 86.5),
        (93, 100),
        (104, 112),
        (120, 129),
        (145, 154),
        (169, 176.8),
    ],
    "135045/iNat1208550": [
        "m",
        (0, 7.5),
        (11, 20),
        (21, 30),
        (33, 41),
        (44.5, 53),
        (58, 66),
        (71, 81),
        (84, 94),
        (94, 104),
        (107, 116.5),
        (120, 130),
    ],
    "135045/iNat1208551": [
        "m",
        (6, 16),
        (15, 52),
        (55, 65),
        (67, 97),
        (101, 111),
        (116, 126),
        (128, 149),
        (147, 161),
        (160, 169),
        (170, 183.5),
        (185, 200),
    ],
    "135045/iNat1208552": [
        "m",
        (0, 13),
        (15, 74),
        (84, 95),
        (95, 116),
        (123, 138),
        (136.5, 148),
        (149, 159),
        (158, 169),
        (171, 183),
        (189, 203),
    ],
    "135045/iNat1208572": [
        "m",
        (0, 30),
        (30, 41),
        (39, 51),
        (57, 71),
        (74, 84.5),
        (86.5, 96.5),
        (97, 107),
        (106, 117.5),
        (118, 136),
        (138, 152),
        (149, 198),
        (197, 212.2),
    ],
    "135045/iNat327127": ["m", (0, 9)],
    "135045/iNat48803": ["m", (0, 8), (20.5, 31), (44, 51.3)],
    "1462711/CSA36371": ["m", (0.9, 7.6)],
    "1462711/CSA36379": ["m", (0.9, 7.7)],
    "1462711/CSA36390": ["m", (0.9, 8)],
    "1462737/CSA36341": ["m", (0.8, 7.7)],
    "1462737/CSA36369": ["m", (0.9, 7.3)],
    "1462737/CSA36380": ["m", (0.9, 7.2)],
    "1462737/CSA36381": ["m", (0.9, 7.2)],
    "1462737/CSA36386": ["m", (0.8, 9.4)],
    "1462737/CSA36391": ["m", (0.8, 7.2)],
    "1462737/CSA36395": ["m", (0.8, 7.2)],
    "1564122/CSA34195": ["m", (0, None)],
    "1564122/CSA34196": ["m", (0, None)],
    "1564122/CSA34197": ["m", (0, None)],
    "1564122/CSA34198": ["m", (0, None)],
    "1564122/CSA34199": ["m", (0, None)],
    "1564122/CSA34200": ["m", (0, None)],
    "21038/iNat297879": ["m", (0, 11)],
    "21038/iNat65519": ["m", (13, 120), (160, 300)],
    "21116/iNat296867": ["m", (0, None)],
    "21116/iNat65520": ["m", (0, 6)],
    "24272/XC882885": ["m", (5, 33), (40, 47), (49, 56)],
    "24272/XC893991": ["m", (0, None)],
    "24272/iNat341657": ["m", (0, None)],
    "24272/iNat715387": ["m", (0, None)],
    "24272/iNat901214": ["m", (0, None)],
    "24272/iNat956047": ["m", (0, None)],
    "24292/CSA34649": ["m", (0.9, 129.2)],
    "24292/CSA34651": ["m", (0.9, 95.3)],
    "24292/CSA35021": ["m", (0, 37.9)],
    "41778/XC959831": ["m", (20, 35), (50, 75), (80, 123), (145, 170)],
    "41778/iNat256586": ["m", (0, None)],
    "42087/iNat155127": ["m", (5, 12)],
    "42087/iNat860016": ["m", (0, None)],
    "42113/iNat55456": ["m", (0, None)],
    "42113/iNat557906": ["m", (0, None)],
    "46010/XC961102": ["m", (0, None)],
    "46010/iNat167113": ["m", (0, None)],
    "46010/iNat247099": ["m", (0, None)],
    "46010/iNat579430": ["m", (0, None)],
    "46010/iNat913511": ["m", (0, None)],
    "47067/iNat1255120": ["m", (0, None)],
    "47067/iNat68676": ["m", (6, 43)],
    "476537/CSA35459": ["m", (0.9, 86.5)],
    "476537/CSA35461": ["m", (0.9, 132.006)],
    "476538/XC926710": ["m", (0, None)],
    "476538/iNat1109247": ["m", (0, None)],
    "476538/iNat955995": ["m", (0, None)],
    "476538/iNat955998": ["m", (0, None)],
    "476538/iNat955999": ["m", (0, None)],
    "523060/CSA34180": ["m", (0, None)],
    "523060/CSA34181": ["m", (0, None)],
    "523060/CSA34182": ["m", (0, None)],
    "523060/CSA34183": ["m", (0, None)],
    "528041/CSA36359": ["m", (0.9, 7.6)],
    "528041/CSA36365": ["m", (0.8, 7.5)],
    "548639/CSA34185": ["m", (0, None)],
    "548639/CSA34186": ["m", (0, None)],
    "548639/CSA34187": ["m", (0, 8), (5, 10)],
    "548639/CSA34188": ["m", (0, None)],
    "548639/CSA34189": ["m", (0, None)],
    "555142/iNat1258897": ["m", (0, None)],
    "555142/iNat1274037": ["m", (0, None)],
    "555142/iNat31004": ["m", (0, 8)],
    "555142/iNat337414": ["m", (0, None)],
    "555142/iNat34199": ["m", (0, None)],
    "555142/iNat761666": ["m", (0, None)],
    "64862/CSA18218": ["m", (4.5, 22), (98, 135), (154, 161), (211, 235), (270, 290)],
    "64862/CSA18222": ["m", (4.1, 30), (70, 95)],
    "65336/iNat1122212": ["m", (0, None)],
    "65336/iNat1193236": ["m", (0, None)],
    "65336/iNat1193440": ["m", (0, None)],
    "65336/iNat36346": ["m", (0, None)],
    "65336/iNat521536": ["m", (0, None)],
    "65336/iNat865751": ["m", (0, None)],
    "65419/iNat296897": ["m", (0, None)],
    "65419/iNat48180": ["m", (0, None)],
    "65419/iNat48209": ["m", (0, None)],
    "65547/XC941286": ["m", (0, None)],
    "65547/iNat1103224": ["m", (0, 12), (11, 16.8)],
    "65547/iNat1108714": ["m", (0, None)],
    "65547/iNat1135442": ["m", (0, None)],
    "65547/iNat296892": ["m", (0, None)],
    "65547/iNat360357": ["m", (0, None)],
    "65547/iNat70565": ["m", (0, None)],
    "66016/XC893981": ["m", (0, None)],
    "66016/iNat14973": ["m", (0, None)],
    "66531/iNat40682": ["m", (0, None)],
    "66531/iNat445441": ["m", (0, None)],
    "66578/iNat223502": ["m", (0, None)],
    "66578/iNat315496": ["m", (0, None)],
    "66893/XC941287": ["m", (0, None)],
    "66893/XC941288": ["m", (0, None)],
    "66893/iNat1109827": ["m", (0, None)],
    "66893/iNat1110476": ["m", (0, None)],
    "66893/iNat42824": ["m", (0, None)],
    "67082/iNat221222": ["m", (0, None)],
    "67082/iNat594579": ["m", (0, None)],
    "714022/CSA34203": ["m", (0, 5.5), (2.5, 11), (8, 17), (17, 25)],
    "714022/CSA34204": ["m", (0, 6), (4, 12), (12, 20), (19, 27), (25, 34), (30, 37)],
    "714022/CSA34205": ["m", (0, 5.2), (5, 14), (15.5, 24), (25, 34), (33, 40)],
    "714022/CSA34206": ["m", (0, 7), (6, 15), (15, 23), (22, 28)],
    "714022/CSA34207": ["m", (0, 5.5), (7, 16), (18, 26), (29, 36), (35.8, 42)],
    "787625/iNat117128": ["m", (0, None)],
    "787625/iNat211195": ["m", (0, None)],
    "787625/iNat400557": ["m", (0, None)],
    "787625/iNat46261": ["m", (0, None)],
    "787625/iNat48805": ["m", (0, None)],
    "787625/iNat600359": ["m", (0, None)],
    "787625/iNat673795": ["m", (0, None)],
    "81930/iNat737012": ["m", (0, None)],
    "81930/iNat761667": ["m", (0, None)],
    "868458/CSA34217": ["m", (0, None)],
    "868458/CSA34218": ["m", (0, None)],
    "868458/CSA34219": ["m", (0, None)],
    "868458/CSA34220": ["m", (0, None)],
    "963335/CSA36372": ["m", (0.9, 8.1)],
    "963335/CSA36374": ["m", (0.9, 8.4)],
    "963335/CSA36375": ["m", (0.8, 8.3)],
    "963335/CSA36377": ["m", (0.9, 7.3)],
    "963335/CSA36393": ["m", (0.9, 7.9)],
    # 'plctan1/XC253746': ['m', (0, None)],  # use pseudo for now
    # 'plctan1/XC364102': ['m', (0, None)],
    # 'plctan1/XC454085': ['m', (0, None)],
    # 'plctan1/XC454405': ['m', (0, None)],
    # 'plctan1/XC455604': ['m', (0, None)],
    # 'plctan1/XC639642': ['m', (0, None)],
    "colcha1/XC337020": ["m", (45, 228)],
    "colcha1/XC532406": ["m", (0, 8)],
    "chbant1/XC315058": ["m", (0, 11), (10, 19)],
    "gybmar/XC9608": ["m", (0, 5), (23, 30), (34, 39)],
    "norscr1/XC146508": [
        "m",
        0,
        6,
        13,
        19,
        26,
        30,
        35,
        42,
        48,
        53,
        61,
        63,
        64,
        69,
        80,
        87,
        99,
        107,
        113,
        118,
        121,
        127,
        132,
        144,
    ],
    "norscr1/XC148047": ["m", 2, 6, 20, 24, 28, 41, 46, 57, 65, 69, 76, 103, 108, 112, 116, (118, 136)],
    "norscr1/XC178590": ["m", 1, 5, 9, 12, 17, 21, 28, 36, 40, 44, 48, 58, 62, 66, 70, 75, 80, 86, 91],
    "norscr1/XC178594": ["m", 2, 5, 10, 17, 25, 31, 40, 44, 50, 55, 63, 68, 74, 80, 91, 98, 103, 108, 112],
    "norscr1/XC178596": ["m", (1, 9), (8, 51.15)],
    "norscr1/iNat31894": ["m", (5, 17.71)],
    "52884/CSA15755": ["m", (9, 17), (25, 35), (33, 50)],
    "52884/CSA18797": [
        "m",
        (0, 28),
        (27, 86),
        (92, 125),
        (125, 135),
        (141, 168),
        (251, 278),
        (287, 310),
        (532, 556),
        (560, 585),
        (606, 614),
    ],
    # -----------------------------------------
    # 9-May
    # -----------------------------------------
    "41970/iNat1015847": ["m", (0, None)],  # not reviewed
    "41970/iNat1015848": ["m", (0, None)],
    "41970/iNat1226073": ["m", (0, None)],
    "41970/iNat327629": ["m", (0, None)],
    "41970/XC564865": ["m", (0, None)],
    "41970/XC564866": ["m", (0, None)],
    "41970/XC564867": ["m", (0, None)],
    "41970/XC564868": ["m", (0, None)],
    "41970/XC564869": ["m", (0, None)],
    "41970/XC564870": ["m", (0, None)],
    "41970/XC564871": ["m", (0, None)],
    "41970/XC564872": ["m", (0, None)],
    "41970/XC564880": ["m", (0, None)],
    "41970/XC564885": ["m", (0, None)],
    "41970/XC564888": ["m", (0, None)],
    "42007/iNat106462": ["m", (0, None)],  # species reviewed
    "42007/iNat1167292": ["m", (0, None)],  # volume is very low
    "42007/iNat1167295": ["m", (0, 12)],
    "42007/iNat1189983": ["m", (0, None)],
    "42007/iNat1241987": ["i", (0, 0)],  # Ignore, puma spotted but not heard
    "42007/iNat141493": ["m", (0, 8)],
    "42007/iNat15105": ["i", (0, 0)],  # Ignore, puma spotted but not heard
    "42007/iNat177093": ["m", (0, None)],
    "42007/iNat464817": ["m", (0, None)],
    "42007/iNat500217": ["m", (0, 42), (42, 70), (73, 80), (85, 93), (92, 100), (104, 111)],
    "42007/iNat606213": ["m", (0, None)],
    "42007/iNat606214": ["m", (0, None)],
    "42007/iNat606215": ["m", (0, None)],
    "42007/iNat841400": ["m", (0, None)],
    "42007/iNat841408": ["m", (0, None)],
    "42007/iNat878627": ["m", (0, None)],
    "42007/iNat968957": ["m", (0, 5.1)],
    "42007/iNat987921": ["m", (0, None)],
    "42007/iNat987929": ["m", (0, 16), (27, None)],
    "46010/960885": ["m", (0, None)],  # not reviewed yet; from additional
    "46010/960886": ["m", (0, None)],
    "66531/982573": ["m", (0, None)],
    "67252/iNat1257838": ["m", (0, None)],
    "67252/iNat933802": ["m", (0, None)],
    "67252/XC882988": ["m", (0, 7), (20, 55)],
    "67252/XC882992": ["m", (0, 42)],
    "67252/XC882993": ["m", (0, 17)],
    "67252/XC882994": ["m", (0, 27)],
    "67252/XC882995": ["m", (0, 23), (24, 40), (42, 61), (69, 79)],
    "67252/XC882996": ["m", (0, 27), (29, 52)],
    "67252/XC882997": ["m", (0, 20)],
    "67252/XC882998": ["m", (0, 62)],
    "67252/XC882999": ["m", (0, 35), (50, 130), (130, 240), (278, 300)],
    "67252/XC883000": ["m", (0, 47)],
    "67252/XC929104": ["m", (0, None)],
    "67252/XC952184": ["m", (0, None)],
    "715170/CSA34510": ["m", (0, 78)],
    "715170/CSA34511": ["m", (0, 58)],
    "715170/CSA34512": ["m", (0, 55), (57, 94)],
    "715170/CSA34513": ["m", (0, 35)],
    "715170/CSA34514": ["m", (0, 58)],
    "715170/CSA34515": ["m", (0, 61)],
    "715170/CSA35799": ["m", (0, 59)],
    "715170/CSA35800": ["m", (0, 55)],
    "715170/CSA35801": ["m", (0, 64)],
    "715170/CSA35802": ["m", (0, 72)],
    "715170/CSA35803": ["m", (0, 69)],
    "715170/CSA35804": ["m", (0, 54)],
    "715170/CSA35805": ["m", (0, 88)],
    "715170/CSA35806": ["m", (0, 55)],
    "715170/CSA35807": ["m", (0, 8), (10, 22), (23, 49)],
    "715170/CSA35808": ["m", (0, 59)],
    "715170/CSA35809": ["m", (0, 61)],
    # 'plctan1/154887': ['m', (0, None)],  # not reviewed; from additional; use pseudo for now
    # 'plctan1/18140': ['m', (0, None)],
    # 'plctan1/18656': ['m', (0, None)],
    # 'plctan1/271605': ['m', (0, None)],
    # 'plctan1/271606': ['m', (0, None)],
    # 'plctan1/271607': ['m', (0, None)],
    # 'plctan1/271608': ['m', (0, None)],
    # 'plctan1/274105': ['m', (0, None)],
    # 'plctan1/31907': ['m', (0, None)],
    # 'plctan1/31908': ['m', (0, None)],
    # 'plctan1/31938': ['m', (0, None)],
    # 'plctan1/83081': ['m', (0, None)],
    # 'plctan1/83082': ['m', (0, None)],
    "sahpar1/iNat51287": ["m", (0, None)],
    "sahpar1/XC116987": ["m", (0, None)],
    "sahpar1/XC149742": ["m", (0, 26)],
    "sahpar1/XC15209": ["m", (0, 6)],
    "sahpar1/XC178654": ["m", (0, 6)],
    "sahpar1/XC24333": ["m", (0, None)],
    "sahpar1/XC254102": ["m", (0, None)],
    "sahpar1/XC26915": ["m", (0, 19.5), (23, 52)],
    "sahpar1/XC26916": ["m", (0, None)],
    "sahpar1/XC26917": ["m", (0, 8), (8, None)],
    "sahpar1/XC354130": ["m", (0, None)],
    "sahpar1/XC354131": ["m", (0, None)],
    "sahpar1/XC402590": ["m", (0, 17)],
    "sahpar1/XC414460": ["m", (0, None)],
    "turvul/iNat1107301": ["m", (0, None)],  # not reviewed;
    "turvul/iNat1275362": ["m", (0, None)],
    "turvul/iNat21647": ["m", (0, None)],
    "turvul/iNat272287": ["m", (0, None)],
    "turvul/iNat292918": ["m", (0, None)],
    "turvul/XC115721": ["m", (0, None)],
    "turvul/XC520287": ["m", (0, None)],
    "turvul/XC520288": ["m", (0, None)],
    "turvul/XC748979": ["m", (0, None)],
    "turvul/XC780516": ["m", (0, None)],
    "turvul/XC904279": ["m", (0, None)],
    "woosto/iNat125203": ["m", (0, None)],
    "woosto/iNat238413": ["m", (0, None)],  # different, maybe incorrect sampling
    "woosto/iNat446108": ["m", (0, None)],  # very low volume
    "woosto/iNat446109": ["m", (0, None)],
    "woosto/iNat640627": ["m", (0, None)],
    "woosto/iNat683978": ["m", (0, None)],
    "woosto/iNat684005": ["m", (0, None)],
    "woosto/iNat684010": ["m", (0, None)],
    "woosto/iNat859775": ["m", (0, None)],
    "woosto/XC131065": ["m", (0, None)],
    "woosto/XC131066": ["m", (0, None)],
    "woosto/XC139206": ["m", (0, None)],
    "woosto/XC149638": ["m", (0, 13), (15, 23)],
    "woosto/XC149640": ["m", (0, None)],
    "woosto/XC161837": ["m", (0, None)],
    "woosto/XC161839": ["m", (0, None)],
    "woosto/XC193137": ["m", (0, None)],
    "woosto/XC193139": ["m", (0, 10), (12, 21), (24, 37)],
    "woosto/XC193143": ["m", (0, None)],
    "woosto/XC196308": ["m", (0, None)],
    "woosto/XC197029": ["m", (0, None)],
    "woosto/XC197030": ["m", (0, None)],
    "woosto/XC412495": ["m", (0, None)],
    "woosto/XC45147": ["m", (0, 8)],
    "woosto/XC52057": ["m", (0, None)],
    "woosto/XC607080": ["m", (4, 10)],
    "woosto/XC789813": ["m", (0, None)],
}


ADDITIONAL_AUDIO_FILTERS = {
    "1139490/2391": ["m", (0, 86)],
    "42113/XC975063": ["m", (0, None)],
    #'66016/vaillanti-escape1': ['m', (0, None)],
    #'66016/vaillanti-escape3': ['m', (0, None)],
    #'66016/vaillanti-escape4': ['m', (0, None)],
    "66578/Pristimantis_bogotensis15": ["m", (8, 22), (39, 51)],
    "868458/2388": ["m", (0, None)],
    # 'turvul/XC381486':
}


class SpeechFilter:
    """Class to filter audio by removing human speech and identifying species singing."""

    def __init__(
        self,
        sing_min_duration=2,
        speech_notes_time=7,
        spech_merge_th=0.3,
        speech_min_duration=2,
        speech_start_th=8,
        th=0.5,
        threads=1,
        sr=32000,
        speech_db_th=-50,
    ):
        self.sing_min_duration = sing_min_duration
        self.speech_notes_time = speech_notes_time
        self.spech_merge_th = spech_merge_th
        self.speech_min_duration = speech_min_duration
        self.speech_start_th = speech_start_th
        self.th = th
        self.sr = sr
        self.speech_db_th = speech_db_th
        self.chunk_len = 0.1
        self.chunk = int(self.chunk_len * self.sr)
        self.device = get_device()
        if self.device == "cpu":
            torch.set_num_threads(threads)
        self.model, (self.get_speech_timestamps, _, _, _, _) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, audio, sr, th=None):
        assert sr == self.sr
        if len(audio.shape) > 1:
            audio = audio[0]
        len_audio = len(audio)

        # Power-based detection
        chunk = self.chunk
        power = audio**2
        pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
        power = np.pad(power, (0, pad)).reshape((-1, chunk)).sum(axis=1)
        power_dB = 10 * np.log10(power)
        x = power_dB - self.speech_db_th
        start, end = 0, len_audio
        intersections = np.where(x[:-1] * x[1:] < 0)[0]
        for s, e in zip(intersections[:-1], intersections[1:]):
            if x[s] < x[s + 1] and (e - s) * self.chunk_len >= self.sing_min_duration:
                start, end = s * chunk, e * chunk
                break
            elif x[s] > x[s + 1] and s * self.chunk_len > self.speech_notes_time:
                start, end = 0, s * chunk
                break

        # Model-based detection
        threshold = th if th is not None else self.th
        inference_audio = audio[start:end].to(self.device)
        speech_timestamps = self.get_speech_timestamps(
            inference_audio, self.model, sampling_rate=self.sr, threshold=threshold
        )
        if len(speech_timestamps) > 0:
            s, e = -1e6, -1e6
            for ts in speech_timestamps:
                if ts["start"] - e < self.spech_merge_th * self.sr:  # Merge
                    e = ts["end"]
                else:
                    s, e = ts["start"], ts["end"]
                duration = (e - s) / self.sr
                start_s = (start + s) / self.sr
                if duration >= self.speech_min_duration or (duration > 0.5 and start_s >= 30):
                    if start_s <= self.speech_start_th:
                        break  # Likely a false positive
                    start, end = start, start + s
                    break
        return start, end


def run_curation_pipeline(filenames, custom_filter=None, speech_filter=None, required_sr=32000):
    """Run the curation pipeline to filter and process audio files."""
    custom_filter = custom_filter or {}
    files_edges = {}
    for filename in tqdm(filenames):
        try:
            audio, sr = torchaudio.load(filename)
            if sr != required_sr:
                resampler = Resample(orig_freq=sr, new_freq=required_sr)
                audio = resampler(audio)
                sr = required_sr
            audio = audio[0]
            id_ = "/".join(os.path.splitext(filename)[0].split("/")[-2:])
            sections = custom_filter.get(id_, None)
            if sections:
                sections = sections[1:] if isinstance(sections[0], str) else sections
                start = int(sections[0][0] * sr)
                end = min(int(sections[-1][1] * sr) if sections[-1][1] is not None else len(audio), len(audio))
            elif speech_filter:
                start, end = speech_filter(audio, sr)
                end = min(end, len(audio))
            if start > 0 or end < len(audio):
                print(f"{filename} changed from ({0}, {len(audio)}) to ({start}, {end})")
            files_edges[id_] = (int(start), int(end))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            files_edges[id_] = (-1, -1)
    return files_edges


def standardize_audio_filters(raw_audio_filters):
    """finalize a dict of audio dilters by converting audio hits into audio sections (start, end),
    merging them when applicable (if the distance to the previous hit <= 5s).
    section for a hit is defined a (hit - BAND, hit + BAND)
    The list sections is prefixed with the type of curation:
        - 'm': the bird is vocalizing in every 5s segment of every section
        - 'a': the bird is not guaranted to vocalize in every 5s segment of every section
        - 'i': the bird is not vocalizing; ignore the audio
    """
    BAND = 4

    audio_filters = dict()
    for id_, raw_sections in raw_audio_filters.items():
        prior_hit = -100
        if raw_sections[0] in ["m", "i"]:
            curation = raw_sections[0]
            start = 1
        else:
            curation = "a"
            start = 0
        sections = [curation]
        for sec in raw_sections[start:]:
            if isinstance(sec, tuple):
                sections.append(sec)
                if sec[1] is not None:
                    prior_hit = sec[1] - 4
            else:
                if prior_hit + 5 >= sec:  # merge with previous section
                    sections[-1] = (sections[-1][0], sec + BAND)
                else:
                    start = max(sec - BAND, 0)
                    end = 5 if start == 0 else sec + BAND
                    sections.append((start, end))
                prior_hit = sec

        audio_filters[id_] = sections
    return audio_filters


def main():
    parser = argparse.ArgumentParser(description="Curate dataset by filtering audio files.")
    parser.add_argument(
        "--file_globs", type=str, nargs="+", required=True, help="List of glob patterns to collect audio files."
    )
    parser.add_argument("--n_threads", type=int, default=1, help="Number of threads for processing.")
    parser.add_argument(
        "--output_json", type=str, required=True, help="Path to save the resulting JSON with file bounds."
    )
    args = parser.parse_args()

    ALL_AUDIO_FILTERS = {**TRAIN_AUDIO_FILTERS, **ADDITIONAL_AUDIO_FILTERS}

    train_audio_filters = standardize_audio_filters(ALL_AUDIO_FILTERS)
    speech_filter = SpeechFilter(threads=args.n_threads)

    all_filenames = []
    for glob_pattern in args.file_globs:
        all_filenames.extend(glob(glob_pattern, recursive=True))

    print(f"Found {len(all_filenames)} audio files matching the patterns")

    all_filenames_bounds = run_curation_pipeline(
        all_filenames, custom_filter=train_audio_filters, speech_filter=speech_filter
    )

    write_json(args.output_json, all_filenames_bounds)
    print(f"Saved curated dataset bounds to {args.output_json}")


if __name__ == "__main__":
    main()
