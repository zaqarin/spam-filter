import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {} # словарь, используемый для преобразования меток в числа
        self.num2label = {} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        cleaned_messages = []
        for i in range(len(self._y)):
            cleaned_message = ''
            for j in range(len(self._y[i])):
                if ord(self._y[i][j]) >=48  and ord(self._y[i][j]) <=57:
                    cleaned_message += self._y[i][j].lower()
                elif ord(self._y[i][j]) >=65  and ord(self._y[i][j]) <=90:
                    cleaned_message += self._y[i][j].lower()
                elif ord(self._y[i][j]) >=97  and ord(self._y[i][j]) <=122:
                    cleaned_message += self._y[i][j].lower()
                elif ord(self._y[i][j]) >= 130 and ord(self._y[i][j]) <= 350:
                    cleaned_message += self._y[i][j].lower()
                else:
                    cleaned_message += ' '
            cleaned_messages.append(cleaned_message)

        for i in range(len(cleaned_messages)):
            cleaned_messages[i] = cleaned_messages[i].replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').strip()

        self._y = np.array(cleaned_messages)

        for i, n in enumerate(list(set(self._x))):
            self.label2num[n] = i
            self.num2label[i] = n

        cleaned_messages1 = [self.label2num[i] for i in self._x]

        self._x = np.array(cleaned_messages1)

        # Конец вашего кода


    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        # Начало вашего кода

        np.random.seed(1)
        indeces = np.arange(0, len(self._x))
        np.random.shuffle(indeces)
        val_indeces = indeces[:round(val*len(self._x))]
        test_indeces = indeces[round(val*len(self._x)):round((val+test)*len(self._x))]
        train_indeces = indeces[round((val+test)*len(self._x)):]

        self.train = (self._x[train_indeces], self._y[train_indeces])
        self.test = (self._x[test_indeces], self._y[test_indeces])
        self.val = (self._x[val_indeces], self._y[val_indeces])
        

    def __str__(self):
        return str([self._x, self._y])

if __name__ == '__main__':
    print(f'Module {__name__} is run')
        # Конец вашего кода
