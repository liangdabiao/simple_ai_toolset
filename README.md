# Python FastAPI AI框架 RAG  Agent
 
# 安装：
## 创建虚拟环境： 
virtualenv venv
## 激活虚拟环境：
#### 在Windows上： 
.\venv\Scripts\activate
#### 在Linux或MacOS上： 
source venv/bin/activate
## 安装依赖：
#### 在虚拟环境中，使用pip命令安装requirements.txt中的依赖。 
pip install -r requirements.txt
#### 退出虚拟环境（如果你使用了虚拟环境）：完成安装后，可以通过以下命令退出虚拟环境： 
deactivate

## 执行启动：
 uvicorn app.main:app --host=0.0.0.0 --port=8080


## 功能介绍：

AI大模型的基本开发框架，适合普通后端程序员，功能类似coze包括：fastapi后端接口，搜索，RAG文档解析和向量化，RPA和爬虫，自定义agent，对接第三方数据接口，mongodb数据库，控制json返回，多模态理解和生成等等

有新的AI功能，会马上跟随潮流，添加上去。


## AI开发应用理念：

1， 理解复杂，处理复杂，生成复杂

2， 尽量把不确定的AI黑盒子转为更加确定性的流程和结果

3， 原型开发是容易的，但是真实可用的AI需要程序员开发