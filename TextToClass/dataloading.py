##########################################################################################
## MIT License                                                                          ##
##                                                                                      ##
## Copyright (c) [2019] [ CarloP95 carlop95@hotmail.it,                                 ##
##                        vitomessi vitomessi93@gmail.com ]                             ##
##                                                                                      ##
##                                                                                      ##
## Permission is hereby granted, free of charge, to any person obtaining a copy         ##
## of this software and associated documentation files (the "Software"), to deal        ##
## in the Software without restriction, including without limitation the rights         ##
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            ##
## copies of the Software, and to permit persons to whom the Software is                ##
## furnished to do so, subject to the following conditions:                             ##
##                                                                                      ##
## The above copyright notice and this permission notice shall be included in all       ##
## copies or substantial portions of the Software.                                      ##
##                                                                                      ##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           ##
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             ##
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          ##
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               ##
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        ##
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        ##
## SOFTWARE.                                                                            ##
##########################################################################################

from torch.utils.data import DataLoader, Dataset, random_split

from string import punctuation

class TextLoader(Dataset):
    """
    TextLoader class that expects a file formatted in the following way:
    <<(Action||Class)Name>><<tab>><<Text>>
    E.g. Biking\tA man is riding a bicycle.

    With this class you must use 0 as your Padding Character.

    Constructor:
    -----------
        path: string
            The path to the file that is formatted as written above.

        dict_file: string
            The path to the file that contains the dict_to_classes mapping.
            The mapping must be as the following line:
            E.g. 1 ApplyEyeMakeup

        item_length: int
            The requested length for each sample.

    Properties:
    ----------
        numClasses: int
            The len operator applied to the actions attribute of this class.
            
    """

    def __init__(self, path, dict_file = None, item_length = 100):
        super(TextLoader, self).__init__()
        self.path           = path
        self.actions        = []
        self.samples        = []
        self.descriptions   = []
        self.item_length    = item_length
        self.dict_file      = dict_file

        try:
            with open(path, 'r+') as fileDataset:

                for line in fileDataset:
                    cleanLine = (line.rstrip('\n\r')).lower()
                    cleanLine = cleanLine.translate(str.maketrans('', '', punctuation))
                    action_description = cleanLine.split('\t')
                    self.actions.append(action_description[0])
                    self.descriptions.append(action_description[1])
                    self.samples.append( ( action_description[1], action_description[0]) )

        except FileNotFoundError as err:
            print(f'File in path {path} was not found.\n{err}')
            exit(1)

        self.actions        = list(dict.fromkeys(self.actions))
        self.allWords       = [word for description in self.descriptions for word in description.split(' ')]
        self.words          = list(dict.fromkeys(self.allWords))
        self.vocabulary     = {self.words[idx]: idx + 1 for idx in range(len(self.words))}

        try:
            self.dict_to_class = self.loadDictFromFile(self.dict_file)

        except (ValueError, FileNotFoundError) as ex:
            if isinstance(ex, FileNotFoundError):
                print(f'File {self.dict_file} was not found. Using default enumeration for classes.')
            
            sorted_actions     = sorted(self.actions)
            self.dict_to_class = {action : index for index, action in enumerate(sorted_actions)}


    @property
    def numClasses(self):
        return len(self.actions)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        print(self.samples[index])
        raise NotImplementedError('Need to implement the padding and converting into tensor.')
        return self.samples[index]

    def loadDictFromFile(self, path):
        if not path:
            raise ValueError(f'Get {path}. Expected a Path.')

        with open(path, 'r+') as file:
            toReturn = {line.split()[1]: line.split()[0] for line in file}

        return toReturn


class DataLoaderFactory:

    def __init__(self, dataset, batch_size = 64, validation = True, train_test_ratio = 0.8, valid_train_ratio = 0.2, 
                        train_shuffle = True, test_shuffle = False, num_workers = 8):

        if not isinstance(dataset, Dataset):
            raise TypeError(f'Expected type {type(Dataset)} but found {type(dataset)}')

        self.validation         = validation
        self.batch_size         = batch_size
        self.train_test_ratio   = train_test_ratio
        self.valid_train_ratio  = valid_train_ratio if self.validation else None
        self.num_workers        = num_workers
        self.dataset            = dataset
        self.train_shuffle      = train_shuffle
        self.valid_shuffle      = train_shuffle
        self.test_shuffle       = test_shuffle
        self.train_num_els      = int( len(dataset) * train_test_ratio )
        self.test_num_els       = len(dataset) - self.train_num_els
        self.valid_num_els      = int(self.train_num_els * self.valid_train_ratio) if self.validation else 0
        self.train_num_els      = self.train_num_els - self.valid_num_els
        self.splits             = [self.train_num_els, self.test_num_els, self.valid_num_els] if self.validation else \
                                                [self.train_num_els, self.test_num_els]

    @property
    def dataloaders(self):
        '''
        Get the DataLoaders for Train, [Validation], Test Datasets in the order.
        The Validation Dataset will not be returned if Validation is set to False.
        '''
        subsets = random_split(self.dataset, self.splits)
        shuffle = [self.train_shuffle, self.test_shuffle, self.valid_shuffle]
        dataloaders = []

        for idx, subset in enumerate(subsets):
            dataloaders.append(DataLoader(subset, batch_size = self.batch_size, shuffle = shuffle[idx], num_workers = self.num_workers))

        try:
            validDataLoader = dataloaders[2]; testDataLoader = dataloaders[1]
            dataloaders[1] = validDataLoader; dataloaders[2] = testDataLoader
        except IndexError:
            pass

        return dataloaders
        


if __name__ == '__main__':
    t = TextLoader('/home/carlo/Documents/Cognitive Computing/Text2VideoGAN/caffe/examples/s2vt/results/dataset_Action_Description.txt')
    factory = DataLoaderFactory(t, batch_size=64)
    train_dataset, valid_dataset, test_dataset = factory.dataloaders
    print(f'Train: {len(train_dataset)}els\nValid: {len(valid_dataset)}els\nTest: {len(test_dataset)}els')

    factory = DataLoaderFactory(t, batch_size=1, validation= False)
    train_dataset, test_dataset = factory.dataloaders
    print(f'Train: {len(train_dataset)}els\nTest: {len(test_dataset)}els')

    t = TextLoader('/home/carlo/Documents/Cognitive Computing/Text2VideoGAN/caffe/examples/s2vt/results/dataset_Action_Description.txt', 
                            '/home/carlo/Documents/Cognitive Computing/Text2VideoGAN/mocogan/ucfTrainTestlist/classInd.txt')

    factory = DataLoaderFactory(t, batch_size=64)
    train_dataset, valid_dataset, test_dataset = factory.dataloaders
    print(f'Train: {len(train_dataset)}els\nValid: {len(valid_dataset)}els\nTest: {len(test_dataset)}els')