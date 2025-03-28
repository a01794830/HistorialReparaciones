import time
# show the result
import numpy as np
import pandas as pd

import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from prompts import DBML_schema_prompt,SQL_assistant_JSON_response_schema

# init the aiplatform package
from google.cloud import aiplatform

# get embeddings for a list of texts
from utils import get_embeddings_wrapper, read_table_as_df

import re
import json

class AssistantsRag:
    def __init__(self,PROJECT_ID,LOCATION):
        
          
        LOCATION = "us-central1"
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        LOCATION = "us-east4"
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Load config for bucket and file path
        with open('config.json') as f:
            data = json.load(f)
        my_index_endpoint_id = data['data']['index_endpoint']
        
        #my_index_endpoint_id = "5903348298273521664"  # @param {type:"string"}
        self.vector_store = aiplatform.MatchingEngineIndexEndpoint(my_index_endpoint_id)
        
        query = """
            SELECT * FROM `pe-fcor-ec-coea-explore-dev.Ferreyros.OTRepuestos`
        """
        OTRepuestos = read_table_as_df(query)

        query = """
            SELECT * FROM `pe-fcor-ec-coea-explore-dev.Ferreyros.OTFallas`
        """
        OTFallas = read_table_as_df(query)

        query = """
            SELECT * FROM `pe-fcor-ec-coea-explore-dev.Ferreyros.OTFiles`
        """
        OTFiles = read_table_as_df(query)

        query = """
            SELECT * FROM `pe-fcor-ec-coea-explore-dev.Ferreyros.OTReparaciones`
        """
        OTReparaciones = read_table_as_df(query)
        
        OTReparaciones["search_text"] = OTReparaciones.apply(
            lambda row: " ".join([f"{col}: {row[col]}" for col in OTReparaciones.columns]),
            axis=1
        )
        OTRepuestos["search_text"] = OTRepuestos.apply(
            lambda row: " ".join([f"{col}: {row[col]}" for col in OTRepuestos.columns]),
            axis=1
        )
        OTFallas["search_text"] = OTFallas.apply(
            lambda row: " ".join([f"{col}: {row[col]}" for col in OTFallas.columns]),
            axis=1
        )
        OTFiles["search_text"] = OTFiles.apply(
            lambda row: " ".join([f"{col}: {row[col]}" for col in OTFiles.columns]),
            axis=1
        )

        # get embeddings for the question titles and add them as "embedding" column
        OTReparaciones['id'] = 'OT_REPARACION_' + OTReparaciones['OT_REPARACION'].astype(str)
        OTFallas['id'] = 'OT_FALLAS_' + OTFallas['OT_REPARACION'].astype(str)
        #OTRepuestos['id'] = 'OT_REPUESTOS_' + OTRepuestos['OT_REPARACION'].astype(str)
        OTFiles['id'] = 'OT_FILES_' + OTFiles['OT_REPARACION'].astype(str)
        
        self.OTReparaciones = OTReparaciones
        self.OTFallas = OTFallas
        self.OTFiles = OTFiles
    
    def test_assistant(self,user_query,model):

        model = GenerativeModel(model_name=model)
        prompt0 = f""" '{user_query}'"""

        contents = [prompt0]
        response = model.generate_content(contents)
        
        return response
    
    def main_assistant(self,user_query,sql_assistant_answer,model):

        model = GenerativeModel(model_name=model)
        prompt0 = f"""
        You are a RAG assistant for a Caterpillar machines work orders database, your job is to recieve user queries and SQL_ASSISTANT answers to formulate an adecuate answer to the user.
        The SQL_ASSISTANT is executes a call to Google Big Query using the user query and returns an structured answer in JSON format.
        

        #INSTRUCTIONS
        1) The user query is : '{user_query}'
        2) The SQL_ASSISTANT answer to user query is : '{sql_assistant_answer}'
        2) Analyze the user query, SQL_ASSISTANT answer and the DBML DATABASE SCHEMA
        3) Write an answer to the user
        
        # DBML DATABASE SCHEMA
        {DBML_schema_prompt()}
        
        Now answer the user:
        """

        contents = [prompt0]
        response = model.generate_content(contents)
        #print(response)
        
        return response.text 
        
    def sql_assistant(self,user_query,model):
        query_embeddings = get_embeddings_wrapper([user_query])
        
        # GET SIMILAR ROWS
        response = self.vector_store.find_neighbors(
            deployed_index_id='embvs_tutorial_deployed_pocing',
            queries=query_embeddings,
            num_neighbors=10,
        )

        query_result = pd.DataFrame()
        for idx, neighbor in enumerate(response[0]):
            id = neighbor.id
            if 'REPARACION' in id:
                similar = self.OTReparaciones.query("id == @id", engine="python")
                query_result = pd.concat([query_result, similar], ignore_index=True)
                #print(f"{neighbor.distance:.4f} {similar}")
            elif 'FALLAS' in id:
                similar = self.OTFallas.query("id == @id", engine="python")
                query_result = pd.concat([query_result, similar], ignore_index=True)
                #print(f"{neighbor.distance:.4f} {similar}")
            elif 'FILES' in id:
                similar = self.OTFiles.query("id == @id", engine="python")
                query_result = pd.concat([query_result, similar], ignore_index=True)
                #print(f"{neighbor.distance:.4f} {similar}")
            else:
                print('NO DATAFRAME MATCH')

        
        model = GenerativeModel(model_name=model)
        prompt0 = f"""
        You are an SQL AI Assistant, your job is to recieve user queries and return a SQL query for Google Big Query that will be executed by code to retrieve information from a database.
        
        #INSTRUCTIONS
        1) The user query is : '{user_query}'
        2) Analyze the user query and the DBML DATABASE SCHEMA
        3) Write a JSON with your SQL answer strictly following the JSON RESPONSE SCHEMA
        
        # DBML DATABASE SCHEMA
        {DBML_schema_prompt()}
        
        #JSON RESPONSE SCHEMA
        {SQL_assistant_JSON_response_schema()}
        
        # EXTRA DATA (this is a similarity search of user query to the database tables, use this as reference to apply the correct SQL filters and joints)
        {str(query_result)}
        
        
        IMPORTANT!
        -You must only return the JSON RESPONSE SCHEMA, DONT INCLUDE ANY EXPLANATIONS OR EXTRA TEXT.
        -Your answer must strictly follow the JSON RESPONSE SCHEMA with all the elements.
        -Remember to use the correct data format when applying filters for dates, numbers, codes ,etc. Checkout the DBML SCHEMA NOTES
        """

        contents = [prompt0]
        response = model.generate_content(contents)
        #print(response)
        
        return response.text 
    
    def eda_assistant(self,main_assistant_answer,sql_assistant_answer,model):
        GENERATION_CONFIG = {
            #"max_output_tokens": 8192,
            "temperature": 0.1,
            #"top_p": 0.95,
        }

        SAFETY_SETTINGS = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH, #BLOCK_ONLY_HIGH
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }    
 

        model = GenerativeModel(model_name=model)
        prompt0 = f"""
        You are an EDA assistant (data visualization), your job is to recieve the outputs of two AI assistants (SQL_ASSISTANT and MAIN_ASSISTANT) and write the code.
        
        SQL_ASSISTANT's output: A structured dataset that is the output query table from the SQL_ASSISTANT.
        MAIN_ASSISTANT's response: A textual explanation providing insights, context, or analysis related to the dataset.
        
        Your task is to:
        -Analyze the dataset provided by SQL_ASSISTANT to determine the best type of visualization.
        -Use the insights from MAIN_ASSISTANT to guide the focus of the visualization.
        -Write Python code using Matplotlib and Seaborn to generate high-quality, publication-ready visualizations.
        -Ensure the plots are clear, labeled properly, and effectively conveys the key insights.
        -Make the visualization dynamic, handling different types of data (e.g., categorical, time-series, distributions, correlations).
        -Only return executable Python code that can be used directly in a Jupyter Notebook.

        IMPORTANT!
        -Your response should ONLY contain Python code, formatted for direct execution in a Jupyter Notebook.

        SQL_ASSISTANT  : {str(sql_assistant_answer)}
        MAIN_ASSISTANT : {str(main_assistant_answer)}
        
        Now generate the python code:
        """

        contents = [prompt0]
        response = model.generate_content(contents,generation_config=GENERATION_CONFIG,safety_settings=SAFETY_SETTINGS,stream=False)
        
        return response.text 

