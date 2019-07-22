from argparse import ArgumentParser
from itertools import product

from tqdm import tqdm
import json

def addCLArguments(parser):
    
    parser.add_argument('--path', default = 'caffe/examples/s2vt/results/dataset_Action_Description.txt', type = str,
                            help= 'Set the relative path to find the file that contains the dataset.')

    return parser


def getCLArguments(parser):

    args = parser.parse_args()

    return {
        'path'          : args.path,
    }


def cleanLine(text):
    return text.rstrip('\n\r')

if __name__ == '__main__':

    clArguments = getCLArguments(addCLArguments(ArgumentParser()))

    with open(clArguments['path'], 'r+') as inputFile:

        action_descriptions = {}

        for line in inputFile:
            cleanedLine = cleanLine(line)
            action, description = cleanedLine.split('\t')

            try:
                isinstance(action_descriptions[action], list)

            except KeyError as _:
                action_descriptions[action] = []

            action_descriptions[action].append(description)

        # Here the dictionary is populated as {SumoWrestling: ["A group of men are dancing on a beach", "...", ...], ...}
        ## 1. Check description duplicates in same classes
        
        duplicates_list = {}

        for key, descriptions in action_descriptions.items():

            for description in descriptions:
                local_occurrencies = 0
                local_occurrencies += descriptions.count(description)

                try:
                    isinstance(duplicates_list[(key, description)], list)
                    continue

                except:
                    duplicates_list[(key, description)] = []

                duplicates_list[(key, description)] = local_occurrencies
            
        ## 2. Sum Local occurrencies to find Global ones

        global_occurrencies = {}

        for (outer_className, outer_description) , outer_counter in duplicates_list.items():
            
            for (inner_className, inner_description) , inner_counter in duplicates_list.items():

                if inner_className == outer_className:
                    continue

                outer_counter += inner_counter if outer_description == inner_description else 0
            
            global_occurrencies[(outer_className, outer_description)] = outer_counter

        print(f'Found {len(global_occurrencies)} entries.')

        ## 3. Delete those that are not duplicates.
        print('Deleting those who are not duplicates...')

        occurrencies_to_check = {}

        for (className, description), num_occurrencies in global_occurrencies.items():

            if num_occurrencies <= 1:
                continue
            
            occurrencies_to_check[(className, description)] = num_occurrencies

        print(f'Found {len(occurrencies_to_check)} occurrencies that needs a check.')

        ## 4. Decide strategy to eliminate duplicates
        toDelete_keys       = []
        notToDelete_keys    = []

        words = None
        with open('TextToClass/included_words.json') as checkWords:
            words = json.load(checkWords)
        print(words)
        for (className, description), num_occurrencies in tqdm(occurrencies_to_check.items()):
            
            if (className, description) in toDelete_keys:
                continue


            print(f'{className} -> {description} -- #global occurrencies {num_occurrencies}\n')
            print(f'-- #local occurrencies {duplicates_list[(className, description)]}')
            

        ## 5. Apply the strategies decided above
        newDataset = {}

        for actionClass, descriptions in action_descriptions.items():
            
            for description in descriptions:

                if not (actionClass, description) in toDelete_keys:

                    try:
                        isinstance(newDataset[actionClass], list)

                    except KeyError as _:
                        newDataset[actionClass] = []

                    newDataset[actionClass].append(description)

        with open('output.txt', 'w') as output:
            for action, descriptions in newDataset.items():
                for description in descriptions:
                    output.write(f'{action}\t{description}\n\r')

            
