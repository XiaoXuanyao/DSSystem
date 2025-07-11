
from debug import *
from openai import OpenAI
import lib
import update
import json



class LLM:
    """
    LLM类，负责加载模型和处理用户询问请求
    """

    def __init__(self):
        self.embedding_model = lib.load_embedding_model()
        debug.log("LLMopts", "加载嵌入模型成功")
        self.summary_tokenizer, self.summary_model = lib.load_summary_model()
        debug.log("LLMopts", "加载摘要模型成功")
        # llm = lib.load_local_llm(r"D:\HLCH\Documents\Homework\DSSystem\DeepSeek-R1-Distill-Qwen-1.5B")
        self.llm: OpenAI = lib.load_deepseek_online()
        debug.log("LLMopts", "加载LLM成功")
        update.update_vector_database(chunk_size=256, chunk_overlap=64)
        debug.log("LLMopts", "向量数据库更新成功")


    def base_query(self, 
            sysmes:str,
            usermes: str,
            assmes: str="",
            n_results: int=3
        ) -> str:
        """
        查询函数，接受问题和参考文献，返回查询结果

        Args:
            sysmes (str): 系统消息，描述LLM的角色和任务
            usermes (str): 用户消息，包含用户的问题
            assmes (str): 助手消息，包含参考文献或上下文信息
        
        Returns:
            str: LLM生成的回复内容
        """
        if assmes is None:
            results = lib.query_message(self.embedding_model, "runs/dataset1", "collection1", usermes, n_results=n_results)
            assmes = lib.document_to_string(results, self.summary_tokenizer, self.summary_model)

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
                messages=[
                {"role": "system", "content": sysmes},
                {"role": "user", "content": usermes},
                {"role": "assistant", "content": assmes},
            ],
            stream=False
        )
        response = response.choices[0].message.content
        return response



    def query(self, 
            question: str,
            history: str,
            n_results: int=3
        ) -> str:
        """
        查询函数，执行一次完整的查询流程，接受问题，返回查询结果

        Args:
            question (str): 用户提出的问题
            history (str): 历史对话记录
            n_results (int): 返回的参考文献数量，默认为3
        """
        # 1. 问题改写
        user_question = question
        question = self.base_query(
            sysmes="你是法务服务智能问答系统的问题改写助手，可以帮助用户更好地表达他们的问题。",
            usermes=f"""请使用简介的话语，将用户的问题改写为更清晰、更具体的问题。\n
例1：
问题：我上个月在某电商平台买了一台破壁机，商家宣传说是“静音设计，噪音低于50分贝”，结果收到货后，声音大得像电钻一样，实测有75分贝，我应该怎么办？
改写后：商家虚假宣传破壁机静音效果（实际75分贝远高于宣传的50分贝），该如何维权索赔？
例2：
问题：我的企业破产了怎么办？
改写后：企业破产后，如何进行债务重组和资产处置？
例3：
问题：重新回答一下上一个问题。
改写后：请根据历史对话重新回答最后一个问题。\n
问题：{question}
""",
            n_results=n_results
        )
        debug.log("LLMopts", "改写后的问题：" + question)

        # 2. 资料关键词生成
        #    例子：我上个月在某电商平台买了一台破壁机，商家宣传说是“静音设计，噪音低于50分贝”，结果收到货后，声音大得像电钻一样，实测有75分贝，我应该怎么办？
        query_key = self.base_query(
            sysmes="你是法务服务智能问答系统的资料查找大师。",
            usermes=f"""请指出下列问题需要查找的资料，要查找的资料不能超过3个，输出以json格式输出。\n
例1：  问题：虚假宣传如何进行消费者维权？  回答：```{{"keywords": ["消费者维权途径", "投诉流程", "虚假宣传认定标准"]}}```
例2：  问题：你今天上午做了什么？  回答：```{{"keywords": []}}```\n
问题：{question}""",
            n_results=n_results
        )[8:-3]
        query_key = json.loads(query_key)["keywords"]
        debug.log("LLMopts", "需要查找的资料：" + str(query_key))

        # 3. 资料查找
        references = "{\nreferences: [\n"
        for key in query_key:
            refmes = lib.document_to_string(lib.query_message(
               self.embedding_model, "runs/dataset1", "collection1", key, n_results=n_results
            ), self.summary_tokenizer, self.summary_model)
            references += refmes
        references += "]\n}\n"
        debug.log("LLMopts", "查找到的资料：" + references)
        debug.log("LLMopts", "历史对话：" + history)

        # 4. 生成回复
        output = self.base_query(
            sysmes="""你是法务服务智能问答系统的AI助理，你富有同情心、知识渊博、耐心负责。
你是一个有感情的AI助理，你会认真倾听用户的问题，理解用户心理，与用户共情。
你非常专业，可以简洁清晰地回答用户，并适当询问用户问题细节，来更好地帮助用户解决问题。
你的参考文献必须只能来自上下文！如果上下文没有提供相关资料，你不可以回复其它内容。
如果资料不足以回答用户的问题，你需要告诉用户你无法回答，并建议用户前往官方网站咨询。
作为公司的AI助理，你需要严格按照参考文献来确保回答准确，不可以凭空杜撰，不可以进行不确定性的回复，不能出现‘可能’等字样。
对任何回答，你都需要在回答末尾添加精确到条的参考文献，告诉用户你的回答是基于哪些资料的。
你不可以回答非法务服务智能问答系统服务无关的问题。
你还需要在回答末尾添加免责声明，告诉用户你的回答仅供参考，详细的法律问题要前往官方网站进行咨询。""",
            usermes=f"""用户问题：{user_question}
法务服务智能问答系统的问题改写助手改写后的问题：{question}
    """,
            assmes=f"参考资料：{references}\n\n历史对话：{history}\n\n",
        )
        debug.log("LLMopts", "生成回复：" + output)
        return output
