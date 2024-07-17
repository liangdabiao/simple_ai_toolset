# Standard library imports
import json
import logging
import time
import os

# Third-party imports
from fastapi import APIRouter,Request
from pydantic import ValidationError
from typing import List
from urllib.parse import quote_plus
from pydantic import BaseModel
import requests
from ..models.blog_models import BlogTitles
# Local imports
from ..services import llm_api as llm, prompts as pr
from ..services.language.llm import LLM, LLMProvider
from ..services.tools.generic_loader import load_content
from ..services.tools.ai_personas import persona_prompts_small
from ..services.tools.serp import search_with_serper_api
from ..services.tools.text_chunker import chunk_by_semantics,chunk_by_paragraphs,chunk_by_max_chunk_size
from ..services.tools.SimplerVectors_core import VectorDatabase
from ..services.language.embeddings import LLM as EmbeddingLLM,EmbeddingsProvider
from ..services.prompts_build.prompt_builder import create_multi_value_prompts
from ..services.tools.agent_class_experimental import Agent
from ..services.tools.json_helpers import convert_pydantic_to_json, extract_json_from_text, validate_json_with_pydantic_model, convert_json_to_pydantic_model
from ..services.tools.rapid_api import RapidAPIClient
import chromadb
import pinecone
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from ..config_loader import config
from ..mongo import get_database

logger = logging.getLogger("AppLogger")
router = APIRouter()


os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = config.get("OPENAI_API_BASE")

""" 简单 测试 """
@router.get("/test/start0")
async def start0(request: Request):
    start_time = time.time()
    max_retries = 5

    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    generated_text = llm_instance.generate_response(prompt="generate a sentence of 5 words")

    print(generated_text)

    return {
        "success": True,
        "message": "Generated test Successfully",
        "result": generated_text
    }


""" 简单gpt生成测试 ，确保json格式返回"""
@router.get("/blog/generate-titles")
async def generate_blog_titles(user_topic: str,request: Request):
    start_time = time.time()
    max_retries = 5

    for retry_count in range(max_retries):
        try:
            if await llm.is_text_flagged(user_topic):
                return {
                    "success": False,
                    "message": "Input Not Allowed",
                    "result": None
                }

            prompt = pr.generate_blog_titles.format(topic=user_topic)
            result : BlogTitles = await llm.generate_with_response_model(prompt,1,BlogTitles)
          
      
            success_result = True
            return {
                "success": True,
                "message": "Generated Titles Successfully",
                "result": result.titles
            }

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                f"Failed during JSON decoding or validation. Retry count: {retry_count + 1}."
            )
            
        except KeyError as e:
            logger.warning(f"Missing key in JSON: {e}")
            
        except Exception as e:
            logger.error(e)
            continue
        finally:
            elapsed_time = time.time() - start_time
            #do soemthing with the elapsed time

    return {
        "success": False,
        "message": f"Failed to generate titles",
        "result": None
    }



""" 文案生成 """
@router.get("/hook_generator")
async def hook_generator(input_topic: str, input_usage: str, request: Request):
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    hook_generator_prompt = """
    as an expert copywriter specialized in hook generation, your task is to 
    analyze the [Provided_Hook_Examples].

    Use the templates that fit most to generate 3 new Hooks 
    for the following topic: {user_input} and Usage in: {usage}. 

    The output should be ONLY valid JSON as follows:
    [
      {{
        "hook_type": "The chosen hook type",
        "hook": "the generatoed hook"
      }},
      {{
        "hook_type": "The chosen hook type",
        "hook": "the generatoed hook
      }},
      {{
        "hook_type": "The chosen hook type",
        "hook": "the generatoed hook"
      }}
    ]

    [Provided_Hook_Examples]:
    "Hook Type,Template,Use In
    Strong sentence,"[Topic] won’t prepare you for [specific aspect].",Social posts, email headlines, short content
    The Intriguing Question,"What’s the [adjective describing difference] difference between [Subject 1] and [Subject 2]?",Video intros, email headlines, social posts
    Fact,"Did you know that [Interesting Fact about a Topic]?",Video intros, email headlines, social posts, short content
    Metaphor,"[Subject] is like [Metaphor]; [Explanation of Metaphor].",Video intros, email headlines, short content
    Story,"[Time Frame], [I/We/Subject] was/were [Situation]. Now, [Current Situation].",Video intros, short content
    Statistical,"Nearly 70% of [Population] experience [Phenomenon] at least once in their lives.",Blog posts, reports, presentations
    Quotation,"[Famous Person] once said, '[Quotation related to Topic]'.",Speeches, essays, social posts
    Challenge,"Most people believe [Common Belief], but [Contradictory Statement].",Debates, persuasive content, op-eds
    Visual Imagery,"Imagine [Vivid Description of a Scenario].",Creative writing, advertising, storytelling
    Call-to-Action,"If you’ve ever [Experience/Desire], then [Action to take].",Marketing content, motivational speeches, campaigns
    Historical Reference,"Back in [Year/Period], [Historical Event] changed the way we think about [Topic].",Educational content, documentaries, historical analyses
    Anecdotal,"Once, [Short Anecdote related to Topic].",Personal blogs, speeches, narrative content
    Humorous,"Why did [Topic] cross the road? To [Punchline].",Social media, entertaining content, ice-breakers
    Controversial Statement,"[Controversial Statement about a Topic].",Debates, opinion pieces, discussion forums
    Rhetorical Question,"Have you ever stopped to think about [Thought-Provoking Question]? ",Speeches, persuasive essays, social posts
    "
    The JSON object:\n\n"""


    input_prompt = hook_generator_prompt.format(user_input=input_topic,usage=input_usage)

    generated_text = llm_instance.generate_response(prompt=input_prompt)

    print(generated_text)

    return {
        "success": True,
        "message": "Generated hooks Successfully",
        "result": generated_text
    }


""" 读网页，分析内容 """
@router.get("/blog_summary")
async def blog_summary(url: str, request: Request):
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    content = load_content(url).content

    summarize_prompt = f"generate a bullet point summary for the following in chinese: {content}"

    generated_text = llm_instance.generate_response(prompt=summarize_prompt)

    print(generated_text)

    return {
        "success": True,
        "message": "blog_summary Successfully",
        "result": generated_text
    }


""" 读落地页，分析和优化建议 """
@router.get("/landing_page")
async def landing_page(landing_page: str, request: Request):

    all_persona_results = analyze_image("https://image.thum.io/get/fullpage/"+landing_page)

    results_string = '\n'.join(all_persona_results)
    final_prompt = f"Act as a Landing Page Expert Analyzer, please checkout the following feedback from different people about a landing page, extract 7-10 unique suggestions, and return them in a list in JSON format. Feedback: {results_string}"
    # 打印 final_prompt
    print(final_prompt)

    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    overall_analysis = llm_instance.generate_response(prompt=final_prompt)
    print(overall_analysis)

    return {
        "success": True,
        "message": "landing_page Successfully",
        "result": overall_analysis
    }


def analyze_image(image_url):
    all_persona_results = []
    anlysis_prompt = pr.anlysis_prompt
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    for persona in persona_prompts_small:
        for title, prompt in persona.items():
            persona_prompt = f"{prompt} {anlysis_prompt}"
            result = llm_instance.analyze_image_basic(image_url, persona_prompt)
            print(result)
            all_persona_results.append(result)

    return all_persona_results


""" youtube分析 """
@router.get("/youtube")
async def youtube(url: str, request: Request):

    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    # url = "https://www.youtube.com/watch?v=LJeZq8MymAs"

    content = load_content(url).content

    summarize_prompt = f"generate a bullet point summary for the following in chinese: {content}"

    generated_text = llm_instance.generate_response(prompt=summarize_prompt)

    print(generated_text)

    return {
        "success": True,
        "message": "youtube_summary Successfully",
        "result": generated_text
    }


""" 基于内容，生成微博 """
@router.get("/weibo")
async def weibo(url: str, request: Request):

    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o") 
    content = load_content(url).content

    convert_to_tweet_prompt = f"""Extract the key info from the following post, 
and Convert to an engaging 280 chars tweet. post content : {content}""" 

    generated_text = llm_instance.generate_response(prompt=convert_to_tweet_prompt)

    print(generated_text)

    return {
        "success": True,
        "message": "weibo build Successfully",
        "result": generated_text
    }


""" 搜索 with serper api"""
@router.get("/search_with_serper")
async def search_with_serper(search_query: str, request: Request):

    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o") 
 

    search_results = search_with_serper_api(query=search_query,num_results=3)
    print(search_results)

    overall_research = ""

    for result in search_results:
        try:
            content = load_content(str(result.URL)).content
            print(content)
        except ValueError as e:
            # 如果处理失败，打印错误信息并继续执行下一个操作
            content = ""
            print(f"无法处理输入：{e}，正在放弃执行，继续下一个操作")

        main_points = llm_instance.generate_response(prompt=f"extract the key points of the following: {content}")
        if main_points is None:
            main_points = ""
        overall_research = overall_research + "\n\n"  + main_points

    print(overall_research)
    user_prompt = f"extract the key information out of the following {overall_research}"

    final_result = llm_instance.generate_response(prompt=user_prompt)

    print(final_result)

    return {
        "success": True,
        "message": "search_with_serper Successfully",
        "result": final_result
    }



""" 搜索 with jina reader api"""
@router.get("/search_with_jina")
async def search_with_jina(search_query: str, request: Request):

    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o") 
    search_query_encoded = quote_plus(search_query)
    sjina = str("https://s.jina.ai/"+search_query_encoded)
    print(sjina)
    response = requests.get(sjina)
    content = ""

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            # 解析响应的JSON数据
            content = response.text
        except json.JSONDecodeError as e:
            # 如果响应不是有效的JSON，打印错误信息
            print(f"Failed to decode JSON: {e}")
    else:
        print(f"Request failed with status code: {response.status_code}")

     
    print(content)
    main_points = llm_instance.generate_response(prompt=f"extract the key information out of the following: {content}")

    print(main_points)

    return {
        "success": True,
        "message": "search_with_jina Successfully",
        "result": main_points
    }




""" 网页内容-》本地向量-》搜索 """
@router.get("/search_with_vector")
async def search_with_vector(user_input: str, request: Request):
    db = VectorDatabase('VDB')

    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    llm_embedding_instance = EmbeddingLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-ada-002")

    url = "https://www.thepaper.cn/newsDetail_forward_1739818"

    content = load_content(url).content
    print(content)


    #chunk the content
    chunks = chunk_by_semantics(text=content,llm_embeddings_instance=llm_embedding_instance,threshold_percentage=95).chunks

    embedded_documents = []

    for chunk in chunks:
        chunk_embedding = llm_embedding_instance.generate_embeddings(user_input=chunk.text,model_name="text-embedding-ada-002")
        embedded_documents.append(chunk_embedding[0].embedding)


    #save all embeds
    for idx, emb in enumerate(embedded_documents):
         db.add_vector(emb, {"doc_id": idx, "vector": chunks[idx].text}, normalize=True)

    db.save_to_disk("db_yt")


    # Embed the user query
    query_embedding = llm_embedding_instance.generate_embeddings(user_input=user_input,model_name="text-embedding-ada-002")
    query_embedding = db.normalize_vector(query_embedding[0].embedding)  # Normalizing the query vector

    # Retrieving the top similar questions and their answers
    results = db.top_cosine_similarity(query_embedding, top_n=1)

    if results:
        top_match = results[0]
        context = top_match[0]["vector"]
        prompt = f"Answer the following question: {user_input} \n Based on this context only: \n" + top_match[0]["vector"]
        answer = llm_instance.generate_response(prompt=prompt)
        print(answer)

    else:
        print("Bot: I'm not sure how to answer that.")

    return {
        "success": True,
        "message": "search_with_vector Successfully",
        "result": answer
    }


""" 网页-》 chromadb向量数据库 -》搜索 """
""" 启动 chroma db : chroma run --path /chroma_db_path """
@router.get("/search_with_chromadb")
async def search_with_chromadb(user_input: str, request: Request):
    # setup Chroma in-memory, for easy prototyping. Can add persistence easily!
    # client = chromadb.Client()
    client = chromadb.PersistentClient(path="D:/Python-Fast-API-Template-main/path")

    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    # collection = client.create_collection("all-my-documents")
    collection = client.get_or_create_collection(name="all-my-documents")

    # Add docs to the collection. Can also update and delete. Row-based API coming soon!
    # collection.add(
    #     documents=["This is tokenization", "we handle tokenization"], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    #     metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
    #     ids=["doc5", "doc6"], # unique for each doc
    # )

    # Query/search 2 most similar results. You can also .get by id
    results = collection.query(
        query_texts=["tokenization"],
        n_results=2,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )

    print(results)

    # 以上是测试 chromadb 能不能链接


    # 下面是 正式工作，向量导入和搜索
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    llm_embedding_instance = EmbeddingLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-ada-002")
    # url = "https://docs.trychroma.com/getting-started"
    url = './data/abc.txt'
    content = load_content(url).content
    print(content)

    #chunk the content
    chunks = chunk_by_max_chunk_size(text=content,max_chunk_size=200).chunks
    collection = client.get_or_create_collection(name="abc6") 
    for idx,chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.text], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            metadatas=[{"doc_id": str(idx), "vector": chunk.text} ], # filter on these!
            ids=[str(idx)], # unique for each doc
        )

    results = collection.query(
        query_texts=[user_input],
        n_results=2,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )

    print(results)

    if results:
        top_match = results['documents'][0]
        context = ' '.join(top_match)
        prompt = f"Answer the following question: {user_input} \n Based on this context only: \n" + context
        print(prompt)
        answer = llm_instance.generate_response(prompt=prompt)
        print(answer)

    else:
        print("Bot: I'm not sure how to answer that.")

    return {
        "success": True,
        "message": "search_with_chromadb Successfully",
        "result": results
    }




"""  prompt提示语模版 tester """
@router.get("/prompt_tester")
async def prompt_tester(url: str, request: Request):
    
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    ## working with multiple value prompts
    multi_value_prompt_template = """Generate 5 titles for a blog about {topic} and {style}"""

    params_list = [
         {"topic": "SEO tips", "style": "catchy"},
         {"topic": "youtube growth", "style": "click baity"},
         {"topic": "email matketing tips", "style": "SEO optimized"}
    ]

    multi_value_prompt = create_multi_value_prompts(multi_value_prompt_template)
    generated_prompts = multi_value_prompt.generate_prompts(params_list)


    for prompt in generated_prompts:
        response = llm_instance.generate_response(prompt=prompt)
        print(response)


#Define a custom tool
def load_content_from_url(url: str):
    """
    Load the page content from a given URL.
    Parameters: url (str)
    """
    content = load_content(url)
    return content.content


"""   ai agent tester """
@router.get("/ai_agent")
async def ai_agent(url: str, request: Request):
    
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    # Create an agent instance
    agent = Agent(LLMProvider.OPENAI, model_name="gpt-4o")

    #add tools
    agent.add_tool(load_content_from_url)

    user_query = """
    generate a consise bullet point summary of the following article: 
    https://learnwithhasan.com/generate-content-ideas-ai/?"""

    # Generate a response
    agent.generate_response(user_query)








"""   llama_parse """
@router.get("/llama_parse")
async def llama_parse(url: str, request: Request):
    nest_asyncio.apply()
    parser = LlamaParse(
        api_key= config.get("LlamaParse"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available 
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=['./data/eee.pdf'], file_extractor=file_extractor).load_data()
    print(documents)

    # 保存 向量数据库 pinecone
    ### 参考 https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/ ###
    # init pinecone
    pc = Pinecone(api_key=config.get("pinecone"))

    # Now do stuff
    if 'quickstart0' not in pc.list_indexes().names():
        pc.create_index(
            name='quickstart0',
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # construct vector store and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=PineconeVectorStore(pc.Index("quickstart0"))
        )

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
    else:
        # 查询 已存的向量数据
        vector_store = PineconeVectorStore(pc.Index("quickstart0"))
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
 

    

    query_engine = index.as_query_engine()
    response = query_engine.query("号码为'95407'的公司名称是 ?")

    print(response)



"""   确保json返回的demo """
@router.get("/blog/generate-titles-json")
async def generate_blog_titles_json(url: str,request: Request):
    start_time = time.time()
    max_retries = 2
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

    for retry_count in range(max_retries):
        try:
            if await llm.is_text_flagged(url):
                return {
                    "success": False,
                    "message": "Input Not Allowed",
                    "result": None
                }

            topic = "AI and SEO"

            base_prompt = f"Generate 5 Titles for a blog post about the following topic: [{topic}]"

            json_model = convert_pydantic_to_json(BlogTitles(titles=['title1', 'title2']))

            optimized_prompt = base_prompt + f'.Please provide a response in a structured JSON format that matches the following model: {json_model}'

            print(optimized_prompt)
            # Generate content using the modified prompt
            generated_text = llm_instance.generate_response(prompt=optimized_prompt)
            print(generated_text)
            # Extract and validate the JSON from the LLM's response
            json_objects = extract_json_from_text(generated_text)

            #validate the response
            validated, errors = validate_json_with_pydantic_model(BlogTitles, json_objects)

            if errors:
                # Handle errors (e.g., log them, raise exception, etc.)
                print("Validation errors occurred:", errors)

            else:
                model_object = convert_json_to_pydantic_model(BlogTitles, json_objects[0])
                #play with json
                for title in model_object.titles:
                    print(title)


            return {
                "success": True,
                "message": "Generated Titles Successfully",
                "result": model_object.titles
            }

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                f"Failed during JSON decoding or validation. Retry count: {retry_count + 1}."
            )
            
        except KeyError as e:
            logger.warning(f"Missing key in JSON: {e}")
            
        except Exception as e:
            logger.error(e)
            continue
        finally:
            elapsed_time = time.time() - start_time
            #do soemthing with the elapsed time

    return {
        "success": False,
        "message": f"Failed to generate titles",
        "result": None
    }




"""   rapid_api demo  搜图"""
@router.get("/rapid_api")
async def rapid_api(keyword: str, request: Request):
    
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    # Create an agent instance
    rapidAPIClient = RapidAPIClient()
    api_url = "https://duckduckgo-image-search.p.rapidapi.com/search/image"
    headers = {
        'x-rapidapi-key': config.get("RAPIDAPI_API_KEY"),
        'x-rapidapi-host': "duckduckgo-image-search.p.rapidapi.com"
    }
    params = {"q":"earth"}
    api_data = rapidAPIClient.call_api( api_url=api_url , method  = 'GET', headers_extra=headers, params=params)
    data_json = api_data
    print(data_json)


"""   mongodb的数据库功能 """
@router.get("/mongodb")
async def mongodb(keyword: str, request: Request):
    try:
        db = await get_database("llm_cache_db")
        cache_collection = db["llm_cache"]
        await cache_collection.insert_one(
            {
                "test": "test",
                "response": "test"
            }
        )

        threshold_date = datetime.now(timezone.utc) - timedelta(
            days=CACHE_DURATION_DAYS
        )
        my_response = await cache_collection.find_one(
            {
                "test": "test",
                "recorded_date": {"$gte": threshold_date},
            },
            projection={"response": 1, "_id": 0},
        )
        results = my_response.get("response") if my_response else None
        print(results)

        return {
            "success": True,
            "message": "search_with_chromadb Successfully",
            "result": results
        }


    except Exception:
        logger.exception(
            f"llm:my_response exception for test "
        )
        return None





"""   图文回复 """
""" 参考：https://www.cnblogs.com/yourenbo/p/18065421 """
@router.get("/rag_pic")
async def rag_pic(question: str, request: Request): 
   
    

    # 保存 向量数据库 pinecone
    ### 参考 https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/ ###
    # init pinecone
    pc = Pinecone(api_key=config.get("pinecone"))

    # Now do stuff
    if 'abc' not in pc.list_indexes().names():
        pc.create_index(
            name='abc',
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # construct vector store and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=PineconeVectorStore(pc.Index("abc"))
        )

        documents = SimpleDirectoryReader(input_files=['./data/abc.txt']).load_data()
        print(documents)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
    else:
        # 查询 已存的向量数据
        vector_store = PineconeVectorStore(pc.Index("abc"))
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
 

    

    query_engine = index.as_query_engine()
    # 可以指定固定回复格式 参考：https://segmentfault.com/a/1190000044329812
    # query_engine = index.as_query_engine(output_cls=BlogTitles) 

    # 定义qa prompt
    qa_prompt_tmpl_str = (
        "上下文信息如下。\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请根据上下文信息而不是先验知识来回答以下的查询。回复格式：请务必把上下文的图片image jpg放在回答的底部作为参考。"
        "作为一个油画艺术人工智能助手，你的回答要尽可能严谨。\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    # 定义refine prompt
    refine_prompt_tmpl_str = (
        "原始查询如下：{query_str}"
        "我们提供了现有答案：{existing_answer}"
        "我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。"
        "------------"
        "{context_msg}"
        "------------"
        "考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。"
        "Refined Answer:"
    )
    refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)

    # 更新查询引擎中的prompt template
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl,
         "response_synthesizer:refine_template": refine_prompt_tmpl}
    )

    response = query_engine.query(question)

    print(response)

    # Get sources 调试和 验证黑盒子里面的运行 RAG过程 参考：https://www.cnblogs.com/yourenbo/p/18058145
    print(response.metadata)
    print(response.source_nodes)
    print(response.get_formatted_sources())




"""   编程技术回复 """
""" 参考：https://www.cnblogs.com/yourenbo/p/18065421 """
@router.get("/ragtech2")
async def ragtech2(question: str, request: Request): 
   
    

    # 保存 向量数据库 pinecone
    ### 参考 https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/ ###
    # init pinecone
    pc = Pinecone(api_key=config.get("pinecone"))

    # Now do stuff
    if 'ragtech2' not in pc.list_indexes().names():
        pc.create_index(
            name='ragtech2',
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # construct vector store and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=PineconeVectorStore(pc.Index("ragtech2"))
        )

        documents = SimpleDirectoryReader(input_files=['./data/ragtech2.txt']).load_data()
       
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
    else:
        # 查询 已存的向量数据
        vector_store = PineconeVectorStore(pc.Index("ragtech2"))
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
 

    

    query_engine = index.as_query_engine()
    # 可以指定固定回复格式 参考：https://segmentfault.com/a/1190000044329812
    # query_engine = index.as_query_engine(output_cls=BlogTitles) 

    # 定义qa prompt
    qa_prompt_tmpl_str = (
        "上下文信息如下。\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请根据上下文信息而不是先验知识来回答以下的查询。回复格式：markdown 。 "
        "作为一个IT工程师人工智能助手，你的回答要尽可能严谨。\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    # 定义refine prompt
    refine_prompt_tmpl_str = (
        "原始查询如下：{query_str}"
        "我们提供了现有答案：{existing_answer}"
        "我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。"
        "------------"
        "{context_msg}"
        "------------"
        "考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。"
        "Refined Answer:"
    )
    refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)

    # 更新查询引擎中的prompt template
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl,
         "response_synthesizer:refine_template": refine_prompt_tmpl}
    )

    response = query_engine.query(question)

    print(response)





"""   软考考试RAG回复 """
""" 参考：https://www.cnblogs.com/yourenbo/p/18065421 """
@router.get("/ruankao")
async def ruankao(question: str, request: Request): 
   
    

    # 保存 向量数据库 pinecone
    ### 参考 https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/ ###
    # init pinecone
    pc = Pinecone(api_key=config.get("pinecone"))

    # Now do stuff
    if 'ruankao' not in pc.list_indexes().names():
        pc.create_index(
            name='ruankao',
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # construct vector store and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=PineconeVectorStore(pc.Index("ruankao"))
        )

        documents = SimpleDirectoryReader(input_files=['./data/ruankao.txt']).load_data()
       
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
    else:
        # 查询 已存的向量数据
        vector_store = PineconeVectorStore(pc.Index("ruankao"))
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
 

    

    query_engine = index.as_query_engine()
    # 可以指定固定回复格式 参考：https://segmentfault.com/a/1190000044329812
    # query_engine = index.as_query_engine(output_cls=BlogTitles) 




    # 定义qa prompt
    qa_prompt_tmpl_str = (
        "## 角色\n"
        "请你扮演中国软件水平考试高级辅导专家，负责用户发送的概念讲解和发送的题目解答。\n"

        "## 技能\n"
        "### 技能1：概念讲解\n"
        "当我发送一些软考概念题目的上下文信息。\n"
        "Step1：根据概括帮我讲解一下相关内容，讲解时尽量通俗易懂，并给出恰当的例子，优先使用 markdown 表格的形式来呈现\n"
        "Step2：出 2 道相关的选择题，在出完题目的最后给出答案和对答案的详细讲解。\n"

        "输出格式为：\n"
        "=====\n"

        "# 一、AI 讲解\n"
        "<概念讲解>\n"

        "# 二、AI 出题\n"
        "### （1）题目\n"
        "<出对应的2道选择题>\n"
        "### （2）答案和解析\n"
        "<所有选择题的答案和解释，每个答案和对应解释放在一起>\n"

        "=====\n"

        "## 要求\n"
        "1 必须使用中文回答我\n"
        "2 解答时，尽量使用通俗易懂的语言\n"
        "3 讲解时，如果有可能尽量给出相关例子\n"
        "4 讲解时，优先考虑使用 markdown 表格的方式呈现，如果出现不同层级的概念，可以将不同层级的概念用不同的表格表示\n"
        "5 给出答案和解析时，每道题的答案和解释要在一起给出，答案的解释需要详尽\n"
        "上下文信息如下。\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请根据上下文信息而不是先验知识来回答以下的查询。回复格式：markdown 。 "
        "作为一个中国软考考试专家人工智能助手，你的回答要尽可能严谨。\n"
        "Query: 关于 {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    # 定义refine prompt
    refine_prompt_tmpl_str = (
        "原始查询如下：{query_str}"
        "我们提供了现有答案：{existing_answer}"
        "我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。"
        "------------"
        "{context_msg}"
        "------------"
        "考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。"
        "Refined Answer:"
    )
    refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)

    # 更新查询引擎中的prompt template
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl,
         "response_synthesizer:refine_template": refine_prompt_tmpl}
    )

    response = query_engine.query(question)

    print(response)

    # Get sources 调试和 验证黑盒子里面的运行 RAG过程 参考：https://www.cnblogs.com/yourenbo/p/18058145
    # print(response.metadata)
    # print(response.source_nodes)
    # print(response.get_formatted_sources())





"""   多tools的智能体 """
#Define a  search tool
def tool_search(keyword: str):
    """
    Search content from a given Keyword.
    Parameters: keyword (str)
    """
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o") 
    search_query_encoded = quote_plus(keyword)
    sjina = str("https://s.jina.ai/"+search_query_encoded)
    print(sjina)
    response = requests.get(sjina)
    content = ""

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            # 解析响应的JSON数据
            content = response.text
        except json.JSONDecodeError as e:
            # 如果响应不是有效的JSON，打印错误信息
            print(f"Failed to decode JSON: {e}")
    else:
        print(f"Request failed with status code: {response.status_code}")

     
    #print(content)
    #main_points = llm_instance.generate_response(prompt=f"extract the key information out of the following: {content}")

    #print(main_points)

    return content



#Define a  什么值得买 tool
def tool_smzdm():
    """
    寻找优惠商品.
    """
    sjina = str("https://r.jina.ai/https://www.smzdm.com/fenlei/jimupincha/#feed-main/") 
    response = requests.get(sjina)
    content = ""

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            # 解析响应的JSON数据
            content = response.text
            parts = content.split("[24小时排行]")
            # 检查是否有分割后的部分
            if len(parts) > 1:
                # 获取分割后的最后一部分
                content = parts[1]
            else:
                print("分隔符不存在于字符串中")
        except json.JSONDecodeError as e:
            # 如果响应不是有效的JSON，打印错误信息
            print(f"Failed to decode JSON: {e}")
    else:
        print(f"Request failed with status code: {response.status_code}")


    return content



#Define a  最新积木新品 tool
def tool_brick4():
    """
    寻找最新商品新品.
    """
    sjina = str("http://brick4.com/get/set?filter_brandorder=0&filter_order=1&brandorder=1&page=1") 
    response = requests.get(sjina)
    content = ""

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            # 解析响应的JSON数据
            content = response.json()
            # 检查'data'字段是否存在，并且它是一个数组
            if 'data' in content and isinstance(content['data'], list):
                # 将数组转换成字符串
                first_6_elements = content['data'][:6]
                content = json.dumps(first_6_elements, ensure_ascii=False)
            else:
                content = ""  # 如果'data'字段不存在或不是数组，使用默认值
            # print(content)
        except json.JSONDecodeError as e:
            # 如果响应不是有效的JSON，打印错误信息
            print(f"Failed to decode JSON: {e}")
    else:
        print(f"Request failed with status code: {response.status_code}")

    return content


#Define a  积木新闻 tool
def tool_brick_news():
    """
    寻找优惠商品.
    """
    sjina = str("https://r.jina.ai/https://zhiyou.smzdm.com/member/6120925173/article/") 
    response = requests.get(sjina)
    content = ""

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            # 解析响应的JSON数据
            content = response.text
            parts = content.split("6120925173/zhuanlan")
            # 检查是否有分割后的部分
            if len(parts) > 1:
                # 获取分割后的最后一部分
                content = parts[1]
                parts2 = content.split(" [下一页]")
                # 检查是否有分割后的部分
                if len(parts2) > 1:
                    # 获取分割后的最后一部分
                    content = parts2[0]
            else:
                print("分隔符不存在于字符串中")
        except json.JSONDecodeError as e:
            # 如果响应不是有效的JSON，打印错误信息
            print(f"Failed to decode JSON: {e}")
    else:
        print(f"Request failed with status code: {response.status_code}")

    return content


"""   ai agent tester """
@router.get("/ai_agent_lego")
async def ai_agent_lego(question: str, request: Request):
    
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    # Create an agent instance
    agent = Agent(LLMProvider.OPENAI, model_name="gpt-4o")

    #add tools
    agent.add_tool(tool_search)
    agent.add_tool(tool_smzdm)
    agent.add_tool(tool_brick4)

    user_query = question

    # Generate a response
    final_result = agent.generate_response(user_query)
    print(final_result)


"""   模仿hackernew 推送和新闻头条 """
@router.get("/ai_write_lego")
async def ai_write_lego(request: Request):
    
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    
    tool_brick_news_str = tool_brick_news()
    main_points_1 = llm_instance.generate_response(prompt=f"keep image and extract the key information out of the following [积木情报]: {tool_brick_news_str}")
   
    tool_smzdm_str = tool_smzdm()
    main_points_2 = llm_instance.generate_response(prompt=f"extract the key information out of the following [积木优惠]: {tool_smzdm_str}")
    tool_brick4_str = tool_brick4()
    main_points_3 = llm_instance.generate_response(prompt=f"extract the key information out of the following [积木新品]: {tool_brick4_str}")
    all_str ="# 今天积木情报: \n\n" + main_points_1 + "\n\n # 今天积木优惠: \n\n" + main_points_2 + "\n\n # 今天积木新品: \n\n" + main_points_3
    
    main_points_best = llm_instance.generate_response(prompt=f"你是一个积木专栏作家，请给上下文内容一个总结，上下文如下: {all_str}")
    print(main_points_best)
    print(all_str)

"""   firecrawl 爬整个网站 /ApifyActor TODO """


"""   连续对话和记忆 """


"""   定时任务 """

"""   oss上传 """


"""   RPA控制浏览器 """
    

    


