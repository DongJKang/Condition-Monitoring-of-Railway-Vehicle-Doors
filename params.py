task1 = {
    'path': './task1',
    'uda_method': 'mcd',
    'source': 'T1',
    'target': 'T2',
    'batch_size': 32,
    'param_e': {
        'seq_len': 70,
        'input_dim': 2,
        'hidden_dim': 20,
        'num_node': 48,
    },
    'param_c': {
        'hidden_dim': 20,
        'num_node': 40,
        'out_node': 4,
    },
}
task2 = {
    'path': './task2',
    'uda_method': 'adda',
    'source': 'T2b',
    'target': 'F',
    'batch_size': 32,
    'param_e': {
        'seq_len': 70,
        'input_dim': 2,
        'hidden_dim': 20,
        'num_node': 48,
    },
    'param_c': {
        'hidden_dim': 20,
        'num_node': 40,
        'out_node': 1,
    },
}