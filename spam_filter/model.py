import numpy as np
import re
from dataset import Dataset

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None # словарь, используемый для преобразования меток в числа
        self.num2label = None # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        self._train_X, self._train_y = dataset.train
        self._val_X, self._val_y = dataset.val
        self._test_X, self._test_y = dataset.test

        self.vocab = {}

        for sentence in self._train_y:
            words = sentence.split()
            for word in words:
                if word in self.vocab:
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1

        for i in range(len(self._train_y)):
            if self._train_X[i] == 1:
                for word in self._train_y[i].split():
                    if word in self.spam:
                        self.spam[word] += 1
                    else:
                        self.spam[word] = 1

        for i in range(len(self._train_y)):
            if  self._train_X[i] == 0:
                for word in self._train_y[i].split():
                    if word in self.ham:
                        self.ham[word] += 1
                    else:
                        self.ham[word] = 1

        self.label2num = dataset.label2num
        self.num2label = dataset.num2label

        self.Nvoc = len(self.vocab)
        self.Nham = len(self.ham)
        self.Nspam = len(set(self.spam))
        # Конец вашего кода

    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        cleaned_message = ''
        for i in range(len(message)):
            if ord(message[i]) >= 48 and ord(message[i]) <= 57:
                cleaned_message += message[i].lower()
            elif ord(message[i]) >= 65 and ord(message[i]) <= 90:
                cleaned_message += message[i].lower()
            elif ord(message[i]) >= 97 and ord(message[i]) <= 122:
                cleaned_message += message[i].lower()
            elif ord(message[i]) >= 130 and ord(message[i]) <= 350:
                cleaned_message += message[i].lower()
            else:
                cleaned_message += ' '

        cleaned_message = cleaned_message.replace('     ', ' ').replace('    ', ' ').replace('   ',
                                                                                                     ' ').replace('  ',
                                                                                                                  ' ').strip()

        p_spam = len(self._train_X[self._train_X==0])/len(self._train_X)
        p_ham = len(self._train_X[self._train_X == 1])/len(self._train_X)

        p_spam_total = 1
        p_ham_total = 1

        cleaned_message = cleaned_message.split()

        for word in cleaned_message:
            if word in self.spam:
                p_spam_total *= self.spam[word] / sum(self.spam.values())
            else:
                p_spam_total *= (self.alpha) / (sum(self.spam.values()) + self.alpha * self.Nvoc)

        for word in cleaned_message:
            if word in self.ham:
                p_ham_total *= self.ham[word] / sum(self.ham.values())
            else:
                p_ham_total *= (self.alpha) / (sum(self.ham.values()) + self.alpha * self.Nvoc)

        p_spam_total *= p_spam
        p_ham_total *= p_ham

        if p_spam_total>p_ham_total:
            return 'spam'
        else:
            return 'ham'

    # Конец вашего кода
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        acc = 0
        for i, message in enumerate(self._val_y):
            inf = self.inference(message)
            inf = self.label2num[inf]
            if self._val_X[i] == inf:
                acc += 1
        val_acc = acc/len(self._val_y)
        # Конец вашего кода
        return f'Validation accuracy: {"%.2f"%val_acc}'

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        acc = 0
        for i, message in enumerate(self._test_y):
            inf = self.inference(message)
            inf = self.label2num[inf]
            if self._test_X[i] == inf:
                acc += 1
        test_acc = acc / len(self._test_y)
        # Конец вашего кода
        return f'Test accuracy: {"%.2f"%test_acc}'
