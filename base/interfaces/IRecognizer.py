'''
IRecognizer interface
定义了推理器的接口

接口说明:
    1、推理器接口，所有推理器都需要实现该接口
    2、推理器接口包含了模型的初始化、推理、释放等方法
    3、推理器需要接收一个存放继承IModel接口的模型的存储路径
    
推理说明:
    1、推理器的推理通过一个进程实现，不断接收数据并进行推理
    2、推理器的推理可以通过start_recognize()和stop_recognize()方法控制
'''
import abc

class IRecognizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _start_recognize(self):
        """
        开始推理，推理的一次过程必须从此开始
        """
        pass

    @abc.abstractmethod
    def _stop_recognize(self):
        """
        结束推理，推理的一次过程必须以此结束
        """
        pass

    @abc.abstractmethod
    def _recognize(self):
        """
        推理数据，具体的推理方式由具体的推理器实现
        """
        pass

    @abc.abstractmethod
    def _parse_data(self, data):
        """
        解析数据，将接收到的数据解析为模型可以接收的数据格式

        Args:
            data (list): 接收到的数据
        """
        pass

    @abc.abstractmethod
    def _parse_result(self, result):
        """
        解析结果，将模型输出的结果解析为接收器可以接收的数据格式

        Args:
            result (list): 模型输出的结果
        """
        pass