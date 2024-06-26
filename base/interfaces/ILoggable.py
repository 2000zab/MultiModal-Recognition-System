'''
ILoggable接口，所有需要记录日志的类都需要实现这个接口

ILoggable接口定义了一个方法，即log()，用于记录日志
'''
from abc import ABCMeta, abstractmethod

class ILoggable(metaclass=ABCMeta):
    @abstractmethod
    def log(self, level, message):
        """
        记录日志

        Args:
            level (str): 日志级别
            message (str): 日志内容
        """
        pass