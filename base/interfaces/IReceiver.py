'''
IReceiver，定义接收器的接口

接口说明：
    1、接收器接口，所有接收器都需要实现该接口，同时继承Process类，实现多进程
    2、接收的数据包括采集的数据和控制trigger两部分，都记录在instance内部
    
数据采集说明：
    1、接收器的数据采集利用一个子线程实现，不断接收数据并存储在instance内部
    2、接收器的数据采集可以通过start_receive()和stop_receive()方法控制
    3、接收器的数据长期保存可以通过store()方法实现
    
控制trigger说明：
    1、接收器的控制trigger记录通过一个子线程实现
    2、子线程接收trigger后，根据trigger的内容执行相应的操作
'''
from abc import ABCMeta, abstractmethod

class IReceiver(metaclass=ABCMeta):
    @abstractmethod
    def _start_receive(self):
        """
        开始接收数据，数据的一次采集必须从此开始
        """
        pass

    @abstractmethod
    def _stop_receive(self):
        """
        结束数据采集，数据的一次采集必须以此结束
        """
        pass

    @abstractmethod
    def _store(self):
        """
        保存数据，数据的一次采集结束后，可以调用此方法保存数据
        具体保存的数据格式由具体的接收器实现
        """
        pass
    
    @abstractmethod
    def _receive(self):
        """
        从数据源接收数据，具体的数据接收方式由具体的接收器实现
        """
        pass
    
    @abstractmethod
    def _parse_trigger(self, trigger):
        """
        设置控制trigger，接收器根据trigger的内容执行相应的操作

        Args:
            trigger (list): 控制trigger的内容
        """
        pass