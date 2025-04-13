import subprocess

import json
import pprint
import os
import shutil

from nn_dataflow.nns import all_networks

import xlsxwriter

workbook_cost = xlsxwriter.Workbook('cost.xlsx')
worksheet_cost = workbook_cost.add_worksheet()
workbook_time = xlsxwriter.Workbook('time.xlsx')
worksheet_time = workbook_time.add_worksheet()
row = 1

# Everything
# BATCH_SIZES = [1, 4, 16, 64, 256]
# NODE_DIMENSIONS = [4, 8, 16, 32]
# ARRAY_DIMENSIONS = [8]
# REGISTER_FILE_SIZES = [64]
# GLOBAL_BUFFER_SIZES = [x * 2**20 for x in [32]]
# MEMORY_TYPES = ['2D', '3D']
# GOALS = ['E', 'D', 'ED']

SYSTEMS = ['M', 'B', 'T']

for system in ['T']:
    match system:
        case 'M':
            # Monolithic
            BATCH_SIZES = [64]
            NODE_DIMENSIONS = [1]
            ARRAY_DIMENSIONS = [128]
            REGISTER_FILE_SIZES = [64]
            GLOBAL_BUFFER_SIZES = [x * 2**20 for x in [8]]
            MEMORY_TYPES = ['2D']
            GOALS = ['E']
        case 'B':
            # Baseline
            BATCH_SIZES = [64]
            NODE_DIMENSIONS = [16]
            ARRAY_DIMENSIONS = [8]
            REGISTER_FILE_SIZES = [64]
            GLOBAL_BUFFER_SIZES = [x * 2**10 for x in [32]]
            MEMORY_TYPES = ['2D'] # Tetris is 3D, does that mean Baseline is 3D? no? maybe?
            GOALS = ['E']
        case 'T':
            # Tangram
            BATCH_SIZES = [64]
            NODE_DIMENSIONS = [16]
            ARRAY_DIMENSIONS = [8]
            REGISTER_FILE_SIZES = [64]
            GLOBAL_BUFFER_SIZES = [x * 2**10 for x in [32]]
            MEMORY_TYPES = ['2D']
            GOALS = ['E']

    dir_path = 'results'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for net, batch, node, array, regf, gbuf, mem_type, goal in [(net, batch, node, array, regf, gbuf, mem_type, goal) 
                                                                for net in
                                                                    ['alex_net', 
                                                                    ]
                                                                for batch in list(map(str, BATCH_SIZES))
                                                                for node in list(map(str, NODE_DIMENSIONS))
                                                                for array in list(map(str, ARRAY_DIMENSIONS))
                                                                for regf in list(map(str, REGISTER_FILE_SIZES))
                                                                for gbuf in list(map(str, GLOBAL_BUFFER_SIZES))
                                                                for mem_type in MEMORY_TYPES
                                                                for goal in GOALS]:
        file_name = '{}-{}-{}-{}-{}-{}-{}-{}-{}.json'.format(system, net, batch, node, array, regf, gbuf, mem_type, goal)
        print(file_name)
        if system=='M':
            subprocess.run(['python3', './nn_dataflow/tools/nn_dataflow_search.py', 
                            net, 
                            '--batch', batch, 
                            '--nodes', node, node, 
                            '--array', array, array, 
                            '--regf', regf, 
                            '--gbuf', gbuf, 
                            '--mem-type', mem_type, 
                            '--goal', goal, 
                            '--bus-width', '8', 
                            # '--unit-idle-cost', '10000', 
                            '--disable-bypass', 'i', 'o', 'f', # tetris does the bypass stuff, should all 3 be disabled for tangram? 
                                                               # for baseline? maybe? default is none so just roll with that with both?
                            # '--solve-loopblocking', # Baseline, incompatible with gbuf-sharing & save-writeback
                            # '--hybrid-partition', # Tetris, so Baseline?
                            # '--batch-partition', # Tetris, so Baseline?
                            # '--ifmaps-partition', # Tetris, so Baseline?
                            # '--enable-access-forwarding', # is implied by below, cannot enable both
                            # '--enable-gbuf-sharing', # Tangram
                            # '--enable-save-writeback', # Tangram?
                            '--disable-interlayer-opt', # Baseline, non-disabled is Tangram, probably?
                            # '--interlayer-partition', # Tangram, but how desirable is it when 
                            #                         # --layer-pipeline-time-overhead & --layer-pipeline-max-degree 
                            #                         # default values is infinity?
                            '--verbose', 
                            ], shell=True)
            shutil.move('result.json', os.path.join(dir_path, file_name))
        elif system=='B':
            subprocess.run(['python3', './nn_dataflow/tools/nn_dataflow_search.py', 
                            net, 
                            '--batch', batch, 
                            '--nodes', node, node, 
                            '--array', array, array, 
                            '--regf', regf, 
                            '--gbuf', gbuf, 
                            '--mem-type', mem_type, 
                            '--goal', goal, 
                            '--bus-width', '8', 
                            # '--unit-idle-cost', '10000', 
                            '--disable-bypass', 'i', 'o', 'f', # tetris does the bypass stuff, should all 3 be disabled for tangram? 
                                                               # for baseline? maybe? default is none so just roll with that with both?
                            # '--solve-loopblocking', # Baseline, incompatible with gbuf-sharing & save-writeback
                            # '--hybrid-partition', # Tetris, so Baseline?
                            # '--batch-partition', # Tetris, so Baseline?
                            # '--ifmaps-partition', # Tetris, so Baseline?
                            # '--enable-access-forwarding', # is implied by below, cannot enable both
                            '--enable-gbuf-sharing', # Tangram
                            '--enable-save-writeback', # Tangram?
                            # '--disable-interlayer-opt', # Baseline, non-disabled is Tangram, probably?
                            '--interlayer-partition', # Tangram, but how desirable is it when 
                                                    # --layer-pipeline-time-overhead & --layer-pipeline-max-degree 
                                                    # default values is infinity?
                            '--verbose', 
                            ], shell=True)
            shutil.move('result.json', os.path.join(dir_path, file_name))
        elif system=='T':
            subprocess.run(['python3', './nn_dataflow/tools/nn_dataflow_search.py', 
                            net, 
                            '--batch', batch, 
                            '--nodes', node, node, 
                            '--array', array, array, 
                            '--regf', regf, 
                            '--gbuf', gbuf, 
                            '--mem-type', mem_type, 
                            '--goal', goal, 
                            '--bus-width', '8', 
                            # '--unit-idle-cost', '10000', 
                            # '--disable-bypass', 'i', 'o', 'f', # tetris does the bypass stuff, should all 3 be disabled for tangram? 
                            #                                    # for baseline? maybe? default is none so just roll with that with both?
                            # '--solve-loopblocking', # Baseline, incompatible with gbuf-sharing & save-writeback
                            # '--hybrid-partition', # Tetris, so Baseline?
                            # '--batch-partition', # Tetris, so Baseline?
                            # '--ifmaps-partition', # Tetris, so Baseline?
                            # '--enable-access-forwarding', # is implied by below, cannot enable both
                            '--enable-gbuf-sharing', # Tangram
                            '--enable-save-writeback', # Tangram?
                            # '--disable-interlayer-opt', # Baseline, non-disabled is Tangram, probably?
                            '--interlayer-partition', # Tangram, but how desirable is it when 
                                                    # --layer-pipeline-time-overhead & --layer-pipeline-max-degree 
                                                    # default values is infinity?
                            '--verbose', 
                            ], shell=True)
            shutil.move('result.json', os.path.join(dir_path, file_name))

        with open(os.path.join(dir_path, file_name), "r") as file:
            str = file.read()
            data = json.loads(str)
            total_cost = data['total_cost']
            pprint.pprint(total_cost)

        #     worksheet_cost.write_row(row, 1, 
        #                                 [data['total_access_cost'], 
        #                                 data['total_noc_cost'], 
        #                                 data['total_op_cost'], 
        #                                 data['total_cost'],data['total_time'],] 
        #                                 )
        #     worksheet_time.write_row(row, 1, 
        #                     [data['total_time'],] 
        #                     )
        #     row += 1

workbook_cost.close()
workbook_time.close()