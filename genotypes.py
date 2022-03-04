from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal_dropped_list normal_concat reduce_dropped_list reduce_concat')

PRIMITIVES = [
    'none',                 #0
    'max_pool_3x3',         #1
    'avg_pool_3x3',         #2
    'skip_connect',         #3
    'sep_conv_3x3',         #4
    'sep_conv_5x5',         #5
    'dil_conv_3x3',         #6
    'dil_conv_5x5'          #7
]

# PRIMITIVES = [
#     'max_pool_3x3',  # 1
#     'avg_pool_3x3',  # 2
#     'skip_connect',  # 3
#     'sep_conv_3x3',  # 4
#     'sep_conv_5x5',  # 5
#     'dil_conv_3x3',  # 6
#     'dil_conv_5x5'  # 7
# ]
