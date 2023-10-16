# Databricks notebook source
import openai 
from pyspark.sql import functions as F
import time

# COMMAND ----------

openai.api_type = 'azure'
openai.api_key = 'a46a37b865a742e2bf138150ff0add57'
openai.api_base = 'https://openaihackathonteamfigo.openai.azure.com/'
openai.api_version = "2023-05-15"

# COMMAND ----------

# DBTITLE 1,Data
df_companies = spark.sql('''SELECT DISTINCT Customer_Name FROM industry_ci WHERE Industry_lvl1 = 'Industry Not Assigned' LIMIT 1000 ''')
                         

df_industries = spark.sql('''SELECT DISTINCT JLL_Industry_Group FROM sic_mapping WHERE JLL_Industry_Group != 'Non Classifiable' ''')
df_sub_industries = spark.sql('''SELECT DISTINCT SIC_Description FROM sic_mapping''')
df_topcompanies = spark.sql('''
WITH df AS (SELECT Customer_Name, SUM(total_amount_usd) 
FROM industry_bi_final_updated
WHERE Customer_Name NOT IN ('NO CLIENT', '-')
GROUP BY Customer_Name
ORDER BY SUM(total_amount_usd) DESC
LIMIT 100) SELECT DISTINCT Customer_Name FROM df                          
''')

# COMMAND ----------

# DBTITLE 1,df to list
company_list = df_companies.select('Customer_Name').rdd.flatMap(lambda x: x).collect()
industry_list = df_industries.select('JLL_Industry_Group').rdd.flatMap(lambda x: x).collect()
sub_industry_list = df_sub_industries.select('SIC_Description').rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# DBTITLE 1,Top Companies
df_topcompanies = df_topcompanies.withColumn("index", F.monotonically_increasing_id())

batch_size = 5
df_topcompanies = df_topcompanies.withColumn("batch_id", (df_topcompanies["index"] / batch_size).cast("integer"))

df_batches = df_topcompanies.groupby("batch_id").agg(F.collect_list("Customer_Name").alias("companies"))

responses = []

while True:
    try:
        if responses:
            start_index = len(responses)
        else:
            start_index = 0

        for row in df_batches.collect()[start_index:]:
            company_list_string = "\n".join(row["companies"])
            prompt_text = f'''   
Instructions: (
I want you to act as a financial analyst. 
Based on the given list of companies: {company_list_string} 
Please produce the following information in this format: !!CompanyA||IndustryA||ConfidenceScoreA|||CompanyB||IndustryB||ConfidenceScoreB|||CompanyC||IndustryC||ConfidenceScoreC!!:

1.Each company's corresponding industry by only choosing from this industry list:{industry_list}

2.the associated confidence score on a scale of 1-100)


Rules: (
1.When facing similar names, for example: Microsoft instead of Microsoft Corporation, or Amazon.com instead of Amazon assume they are the same company.
2.If a company has businesses in multiple industries, choose the industry that reflecs the company's base/core.
  For example, even tho facebook is a social media company, the way they make money is through technology. Or Amazon is a E-Commmerce comapny but they are really a technology company.
3.Adhere strictly to the prescribed format as one continuous string: for instance, the result should mimic this format: !!CompanyA||IndustryA||ConfidenceScoreA|||CompanyB||IndustryB||ConfidenceScoreB|||CompanyC||IndustryC||ConfidenceScoreC!!
'!!' is used to wrap the begin and end of the string.
4.Do NOT use any other formatting including but not limited to: indexes, indents, spaces, headers, new lines, empty results
5.Do NOT omit or truncate records, produce all results.)
'''

            prompt = [{"role": "user", "content": prompt_text}]
            response = openai.ChatCompletion.create(deployment_id='gpt-35-turbo-16k', engine="gpt-35-turbo-16k", messages=prompt, temperature = 0)
            
            gpt_answer = response["choices"][0].message.content.strip()
            
            companies_industries_scores = gpt_answer.strip().split('|||')
            
            for item in companies_industries_scores:
                item = item.replace('!!', '')
                try:
                    company, industry, confidence_score = map(str.strip, item.split('||'))
                    responses.append((company, industry, confidence_score))
                except ValueError as ve:
                    print(f"Error parsing item {item}: {ve}")

        header = ["Company", "Industry", "Confidence_Score"]
        sdf = spark.createDataFrame(data=responses, schema=header)
        sdf = sdf.withColumn("Confidence_Score", sdf["Confidence_Score"].cast("integer"))

        sdf.createOrReplaceTempView('GPT_topIndustry')
        break

    except Exception as e:  
        print(f"Error: {e}")
        time.sleep(3)

# COMMAND ----------

# DBTITLE 1,TC Sample Data
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM GPT_topIndustry A
# MAGIC LEFT JOIN industry_ci B
# MAGIC ON A.Company = B.Customer_Name
# MAGIC WHERE A.Industry = B.Industry_lvl1

# COMMAND ----------

# DBTITLE 1,Industry
df_companies = df_companies.withColumn("index", F.monotonically_increasing_id())
batch_size = 100
df_companies = df_companies.withColumn("batch_id", (df_companies["index"] / batch_size).cast("integer"))
df_batches = df_companies.groupby("batch_id").agg(F.collect_list("Customer_Name").alias("companies"))

responses = []

while True:
    try:
        #If already at least one response exists, the process will continue after the last successfully processed batch
        if responses:
            start_index = len(responses)
        else:
            start_index = 0

        for row in df_batches.collect()[start_index:]:
            company_list_string = "\n".join(row["companies"])
            prompt_text = f'''   
Instructions: (
I want you to act as a financial analyst. 
Based on the given list of companies: {company_list_string} 
Please produce the following information in this format: !!CompanyA||IndustryA||ConfidenceScoreA|||CompanyB||IndustryB||ConfidenceScoreB|||CompanyC||IndustryC||ConfidenceScoreC!!:

1.Each company's corresponding industry by only choosing from this industry list:{industry_list}

2.the associated confidence score on a scale of 1-100)


Rules: (
1.When facing similar names, for example: Microsoft instead of Microsoft Corporation, or Amazon.com instead of Amazon assume they are the same company.
2.If a company has businesses in multiple industries, choose the industry that reflecs the company's base/core.
  For example, even tho facebook is a social media company, the way they make money is through technology. Or Amazon is a E-Commmerce comapny but they are really a technology company.
3.Adhere strictly to the prescribed format as one continuous string: for instance, the result should mimic this format: !!CompanyA||IndustryA||ConfidenceScoreA|||CompanyB||IndustryB||ConfidenceScoreB|||CompanyC||IndustryC||ConfidenceScoreC!!
'!!' is used to wrap the begin and end of the string.
4.Do NOT use any other formatting including but not limited to: indexes, indents, spaces, headers, new lines, empty results)
5.Do NOT omit or truncate records, produce all results.
'''

            prompt = [{"role": "user", "content": prompt_text}]
            response = openai.ChatCompletion.create(deployment_id='gpt-35-turbo-16k', engine="gpt-35-turbo-16k", messages=prompt, temperature = 0)
            
            gpt_answer = response["choices"][0].message.content.strip()
            
            companies_industries_scores = gpt_answer.strip().split('|||')
            
            for item in companies_industries_scores:
                item = item.replace('!!', '')
                try:
                    company, industry, confidence_score = map(str.strip, item.split('||'))
                    responses.append((company, industry, confidence_score))
                except ValueError as ve:
                    print(f"Error parsing item {item}: {ve}")

        header = ["Company", "Industry", "Confidence_Score"]
        sdf = spark.createDataFrame(data=responses, schema=header)
        sdf = sdf.withColumn("Confidence_Score", sdf["Confidence_Score"].cast("integer"))

        sdf.createOrReplaceTempView('GPT_Industry')
        break

    except Exception as e:  
        print(f"Error: {e}")
        time.sleep(3)

# COMMAND ----------

# DBTITLE 1,Industry Sample Data
# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM GPT_Industry A
# MAGIC LEFT JOIN industry_ci B
# MAGIC ON A.Company = B.Customer_Name

# COMMAND ----------

# DBTITLE 1,Forbes
df_topcompanies = df_topcompanies.withColumn("index", F.monotonically_increasing_id())

batch_size = 50
df_topcompanies = df_topcompanies.withColumn("batch_id", (df_topcompanies["index"] / batch_size).cast("integer"))

df_batches = df_topcompanies.groupby("batch_id").agg(F.collect_list("Customer_Name").alias("companies"))

responses = []

for row in df_batches.collect():
    company_list_string = "\n".join(row["companies"])

    prompt_text = f'''   
Instructions: (
Based on the given list of companies: {company_list_string} 
Please:
1. classify whether they are in the forbes global 2000 list with a Yes/No
2. produce the company's head quarter, put unkown if null
3. produce the company's net worth





Rules: (
1.Make sure to classify all companies.
2.When facing similar names, for example: Microsoft instead of Microsoft Corporation, assume Microsoft is Microsoft Corporation.
3.Adhere strictly to the prescribed format as one continuous string: for instance, the result should mimic this format: "!!CompanyA||Yes/No||headquarterA||revenueA|||CompanyB||Yes/No||headquarterA||revenueA|||CompanyC||Yes/No||headquarterA||revenueA!!".
4.Do NOT use any other formatting including but not limited to: indexes, indents, spaces, headers, new lines.)
'''

    prompt = [{"role": "user", "content": prompt_text}]
    response = openai.ChatCompletion.create(deployment_id='gpt-35-turbo-16k', engine="gpt-35-turbo-16k", messages=prompt, temperature = 0)
    gpt_answer = response["choices"][0].message.content.strip()
    
    companies_industries_scores = gpt_answer.strip().split('|||')
    
    for item in companies_industries_scores:
        item = item.replace('!!', '')
        try:
            company, Forbes, headquarter, revenue = map(str.strip, item.split('||'))
            responses.append((company, Forbes, headquarter, revenue))
        except ValueError as ve:
            print(f"Error parsing item {item}: {ve}")

header = ["Company", "Forbes", "Headquarter", "Net Worth"]
sdf = spark.createDataFrame(data=responses, schema=header)
#sdf = sdf.withColumn("Confidence_Score", sdf["Confidence_Score"].cast("integer"))

sdf.createOrReplaceTempView('GPT_Forbes')

# COMMAND ----------

# DBTITLE 1,Forbes Sample Data
# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM GPT_Forbes
# MAGIC WHERE Forbes = 'Yes'

# COMMAND ----------


