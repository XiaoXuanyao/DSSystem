
from debug import *
import streamlit as st
import httpx
import time



# 页面标题和布局
st.set_page_config(page_title="法务服务智能问答系统", layout="centered")

st.sidebar.title("法务服务智能问答系统")
st.sidebar.markdown("""
> 基于 Deepseek 的法务服务智能问答系统，提供专业的法律问题咨询服务。
                    
**我是法务服务智能问答系统的AI助理。关于如何咨询法律问题，我可以为您提供以下建议：**

您可以详细描述您遇到的具体法律问题，包括：\n
涉及的法律领域（如劳动纠纷、合同问题等）、事件发生的时间、地点和相关人员、您目前掌握的证据材料、您希望达到的解决目标\n
咨询前的准备：\n
整理好相关证据材料、明确要咨询的具体问题、记录事件时间线\n
您能否告诉我您具体想咨询哪方面的法律问题呢？这样我可以为您提供更有针对性的建议。\n

> 免责声明：以上回答仅供参考，具体法律问题建议咨询专业律师或前往当地法律服务机构获取正式法律意见。
""")

chat_box = st.container()

#
#初始化对话历史和查询历史
#
init_label = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['history'] = []
    st.session_state['sendOK'] = True
    with chat_box:
        init_label = st.title("您好，我能帮什么忙？")

for message in st.session_state['messages']:
    if message['role'] in ['user', 'assistant', 'system']:
        with chat_box:
            with st.chat_message(message['role']):
                st.write(message['content'])



#
#用户输入及响应
#
user_input = st.chat_input("您有什么想问的？", max_chars=2000)

if user_input:
    if st.session_state['sendOK'] is False:
        warnmes = "已自动为您停止上一个回答的生成"
        with chat_box:
            with st.chat_message("system"):
                st.session_state.messages.append({"role": "system", "content": warnmes})
                st.write(warnmes)
    st.session_state['sendOK'] = False
    if init_label is not None:
        init_label.empty()
        init_label = None
    with chat_box:
        with st.chat_message("user"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.write(user_input)

    # 再调用 LLM
    with chat_box:
        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                response = httpx.post(
                    "http://localhost:25566/query",
                    json={
                        "question": user_input,
                        "history": f"{{\n{st.session_state['history']}\n}}"
                    },
                    timeout=300.0
                ).json()["response"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state['history'].append({
                "role": "user",
                "content": user_input,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })
            st.session_state['history'].append({
                "role": "assistant",
                "content": response,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })
            st.session_state['history'] = st.session_state['history'][-7:]
            st.session_state['sendOK'] = True
            st.rerun() # 必须！可以防止直接渲染回答导致回答显示两次