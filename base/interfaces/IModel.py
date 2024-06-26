'''
推理模型接口

接口说明：
    1. 模型接口，所有模型都需要实现该接口
    2. 模型接口包含了模型的初始化、推理、释放等方法
    3. 模型需要保证可以被pickle序列化，以便于存储和加载
    4、所有模型都要继承Callable类，实现__call__方法，以便于直接调用模型进行推理
'''
from abc import ABCMeta, abstractmethod
from typing import Any

class IModel(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, data):
        """
        数据预处理

        Args:
            data (list): 输入数据

        Returns:
            list: 预处理后的数据
        """
        pass
    
    @abstractmethod
    def postprocess(self, data):
        """
        数据后处理

        Args:
            data (list): 输入数据

        Returns:
            list: 后处理后的数据
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        模型推理

        Args:
            data (list): 输入数据

        Returns:
            list: 模型输出结果
        """
        pass
    
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass