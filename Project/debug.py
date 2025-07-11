
import time
import os



class costTime:
    """
    计时，用于测试代码块的执行时间
    使用with包裹要测试的代码块
    """

    name: str

    def __init__(self,
            name
        ):
        """
        初始化计时器
        """
        self.name = name

    def __enter__(self):
        """
        进入计时上下文，记录开始时间
        """
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出计时上下文，计算并输出执行时间
        """
        end_time = time.perf_counter()
        count = end_time - self.start_time
        debug.log('Debug', f'{self.name} {count // 60:.0f} min {count - count // 60 * 60:.3f} s')



class debug:
    """
    调试类，提供日志记录功能
    """

    @staticmethod
    def log(
            tag: str,
            mes: str,
            type: int=0,
            end ='\n'
        ) -> None:
        """
        输出日志信息到控制台，记录日志文件

        Args:
            tag (str): 日志标签，用于标识日志来源
            mes (str): 日志消息内容
            type (int): 日志类型，0表示记录到文件，1表示仅输出
            end (str): 输出结束符，默认为换行符
        """
        output = "\033[36m[\033[1m" + time.strftime('%H:%M:%S', time.localtime(time.time())) + "] (" + tag + ") "
        output += "\033[0m\033[37m" + mes + "\033[0m"
        output = output.encode('gbk', errors='ignore').decode('gbk')
        print(output, end=end)
        output = "[" + time.strftime('%H:%M:%S', time.localtime(time.time())) + "] (" + tag + ") "
        output += mes
        output = output.encode('gbk', errors='ignore').decode('gbk')
        if type == 0:
            os.makedirs('runs', exist_ok=True)
            with open(f'runs/log.txt', 'a') as f:
                f.write(output + "\n")
    
    @staticmethod
    def arrayToStr(mes) -> None:
        """
        将数组转换为字符串格式，便于输出
        
        Args:
            mes (list): 要转换的数组
        """
        output = "[ "
        for e in mes:
            output += "{:3}".format(e) + ", "
        return output + "]"