'''
IBackend，实验后端接口，负责实验的后端通信

接口说明：
    1、实验后端接口，所有实验后端都需要实现该接口
    2、实验后端接口包括实验基本信息的记录、实验前端和后端的通信等方法
    3、不需要单开一个进程，直接在主进程中调用即可
    4、在和后端进程通信时，需要使用多进程的Queue进行通信
    5、通信过程使用list类型数据，数据格式为[command, [data]]，command为命令，data为数据
    6、通信过程是可以离开的，即可以在任何时候调用，不会影响实验的进行
'''
from abc import ABCMeta, abstractmethod

class IBackend(metaclass=ABCMeta):
    @abstractmethod
    def record(self, info):
        """
        记录实验信息

        Args:
            info (dict): 实验信息
        """
        pass

    @abstractmethod
    def sendToBackend(self, data):
        """
        与实验前端通信

        Args:
            data (list): 通信数据
        """
        pass
    
    @abstractmethod
    def receiveFromBackend(self):
        """
        从实验前端接收数据

        Returns:
            list: 通信数据
        """
        pass
    
    @abstractmethod
    def startExperiment(self, **kwargs):
        """
        开始实验
        """
        pass
    
    @abstractmethod
    def stopExperiment(self):
        """
        结束实验
        """
        pass