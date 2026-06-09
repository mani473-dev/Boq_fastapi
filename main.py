import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
import operator
from typing import TypedDict, Literal, Annotated


from pandasql import sqldf
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_oci.chat_models import ChatOCIGenAI


#user_input = input("Enter the query: ")

load_dotenv()

BASE_URL = os.getenv("BaseURL")

USERNAME = os.getenv("IntermassUsername")
PASSWORD = os.getenv("IntermassPassword")

CONFIG = os.getenv("ConfigPath")


llm_intent = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaa2jjkoqmd23eccvazv4u6dx746sx7ltkytcxb7swfjibcqdvw6blq",
    auth_profile="DEFAULT",
    auth_file_location= CONFIG, 
    model_kwargs={
        
        "max_tokens": 3000
    },
)

query_generator = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaa2jjkoqmd23eccvazv4u6dx746sx7ltkytcxb7swfjibcqdvw6blq",
    auth_profile="DEFAULT",
    auth_file_location= CONFIG, 
    model_kwargs={
        
        "max_tokens":3000
    },
)

query_evaluator=  ChatOCIGenAI(
    model_id="openai.gpt-oss-120b",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaa2jjkoqmd23eccvazv4u6dx746sx7ltkytcxb7swfjibcqdvw6blq",
    auth_profile="DEFAULT",
    auth_file_location= CONFIG, 
    model_kwargs={
      
        "max_completion_tokens": 3000
    },
)

query_optimizer=  ChatOCIGenAI(
    model_id="google.gemini-2.5-pro",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaa2jjkoqmd23eccvazv4u6dx746sx7ltkytcxb7swfjibcqdvw6blq",
    auth_profile="DEFAULT",
    auth_file_location= CONFIG, 
    model_kwargs={
       
        "max_tokens": 3000
    },
)

result_summarizer_llm=  ChatOCIGenAI(
    model_id="openai.gpt-oss-120b",
    service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaa2jjkoqmd23eccvazv4u6dx746sx7ltkytcxb7swfjibcqdvw6blq",
    auth_profile="DEFAULT",
    auth_file_location= CONFIG, 
    model_kwargs={
      
        "max_completion_tokens": 5000
    },
)


LIMIT = 500
MAX_WORKERS = 8

#dashboard_context = {
#   "project": {
#     "actualPercent": "50.70",
#     "EstimatedAmount": "171800000",
#     "budget": "72.45M",
#     "cost": "36.73M",
#     "CPI": "1.00",
#     "CV": "0",
#     "EAC": "79.82M",
#     "earnedValue": "36.73M",
#     "plannedPercent": "56.3",
#     "plannedValue": "91200000",
#     "profitVariance": "-9800000",
#     "projectName": "112-ASD2",
#     "ProjectId": "300000004277391",
#     "ProjectManagerName": "Integration User",
#     "ProjectNumber": "112-P1131",
#     "ProjectStartDate": "2023-06-07",
#     "SPI": "1.25",
#     "SV": "7.37M"
#   },
#   "periodData": [
#     {
#       "period": "Jan",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Feb",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Mar",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Apr",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "May",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Jun",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Jul",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Aug",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Sep",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Oct",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Nov",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Dec",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Jan",
#       "type": "cost",
#       "cost": -366721.62
#     },
#     {
#       "period": "Feb",
#       "type": "cost",
#       "cost": 14873.03
#     },
#     {
#       "period": "Mar",
#       "type": "cost",
#       "cost": 160234.75
#     },
#     {
#       "period": "Apr",
#       "type": "cost",
#       "cost": 16958.45
#     },
#     {
#       "period": "May",
#       "type": "cost",
#       "cost": 174655.44
#     },
#     {
#       "period": "Jun",
#       "type": "cost",
#       "cost": 0
#     },
#     {
#       "period": "Jul",
#       "type": "cost",
#       "cost": 898030
#     },
#     {
#       "period": "Aug",
#       "type": "cost",
#       "cost": 63530.229999999996
#     },
#     {
#       "period": "Sep",
#       "type": "cost",
#       "cost": -16580.86
#     },
#     {
#       "period": "Oct",
#       "type": "cost",
#       "cost": 0
#     },
#     {
#       "period": "Nov",
#       "type": "cost",
#       "cost": 0
#     },
#     {
#       "period": "Dec",
#       "type": "cost",
#       "cost": 24300
#     }
#   ],
#   "categoryData": [
#     {
#       "Budget": "0",
#       "Cost": "2.25",
#       "Percentage": 0,
#       "ResourceClassName": "Labor"
#     },
#     {
#       "Budget": "0",
#       "Cost": "2.03M",
#       "Percentage": 0,
#       "ResourceClassName": "Equipment"
#     },
#     {
#       "Budget": "0",
#       "Cost": "6.11M",
#       "Percentage": 0,
#       "ResourceClassName": "Material Items"
#     },
#     {
#       "Budget": "3.13M",
#       "Cost": "15.74M",
#       "Percentage": 503.51,
#       "ResourceClassName": "Financial Resources"
#     }
#   ]
# }

# 2. Route Decision Model
class RouteDecision(BaseModel):
    """
    Selects which route should answer the question.
    """
    route: Literal["SCREEN_ROUTE", "DATABASE_ROUTE", "HYBRID_ROUTE"] = Field(
        description="The data source route selected based on the user question."
    )

class QueryState(TypedDict):
    question: str
    investigation_plan: str  # Fixed: Changed = to :
    route: Literal["SCREEN_ROUTE", "DATABASE_ROUTE", "HYBRID_ROUTE"]
    dashboard_context: dict     # Kept as dict/str depending on your data structure
    query: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    sql_result: str
    result_summary: str
    query_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


COLUMN_MAPPING = {
    "TransactionNumber_c": "TransactionNumber",
    "RawCostInProjectCurrency_c": "RawCostInProjectCurrency",
    "Quantity_c": "Quantity",
    "TaskId_c": "TaskId",
    "ProjectNumber_c": "ProjectNumber",
    "ProjectName_c": "ProjectName",
    "TaskNumber_c": "TaskNumber",
    "TaskName_c": "TaskName",
    "ExpenditureCategory_c": "ExpenditureCategory",
    "ExpenditureType_c": "ExpenditureType",
    "AccountingPeriod_c": "AccountingPeriod",
    "ProjectId_c": "ProjectId",
    "Budget_RawCostAmounts_c": "Budget_RawCostAmounts",
    "Budget_RevenueAmounts_c": "Budget_RevenueAmounts",
    "Cost_RawCostInProjectCurrency_c": "Cost_RawCostInProjectCurrency",
    "Variance_c": "Variance"
}


def create_session():
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)
    session.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    return session


def fetch_page(offset):
    session = create_session()

    params = {
        "limit": LIMIT,
        "offset": offset,
        "totalResults": "true"
    }

    response = session.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()

    return response.json()


def get_total_results():
    session = create_session()

    params = {
        "limit": 1,
        "offset": 0,
        "totalResults": "true"
    }

    response = session.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data.get("totalResults", 0)


def get_all_records():
    total_results = get_total_results()
    #print("Total records:", total_results)

    offsets = list(range(0, total_results, LIMIT))
    all_records = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_offset = {
            executor.submit(fetch_page, offset): offset
            for offset in offsets
        }

        for future in as_completed(future_to_offset):
            offset = future_to_offset[future]

            try:
                data = future.result()
                items = data.get("items", [])
                all_records.extend(items)

                print(f"Fetched offset {offset}, records: {len(items)}")

            except Exception as e:
                print(f"Error fetching offset {offset}: {e}")

    #print("Fetched records:", len(all_records))
    #print("Time taken:", round(time.time() - start_time, 2), "seconds")

    return all_records


def create_clean_dataframe(records):
    clean_records = []

    for item in records:
        row = {}

        for source_col, target_col in COLUMN_MAPPING.items():
            row[target_col] = item.get(source_col)

        clean_records.append(row)

    df = pd.DataFrame(clean_records)

    numeric_cols = [
        "RawCostInProjectCurrency",
        "Quantity",
        "Budget_RawCostAmounts",
        "Budget_RevenueAmounts",
        "Cost_RawCostInProjectCurrency",
        "Variance"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


records = get_all_records()

df = create_clean_dataframe(records)


# -----------------------------
# Helpers
# -----------------------------
def clean_sql(text: str) -> str:
    text = text.strip()

    if "```sql" in text:
        return text.split("```sql")[1].split("```")[0].strip()

    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()

    return text.strip()


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError("Could not parse JSON from evaluator response")




def intent_router(state: QueryState):

   
    context_data = state.get("dashboard_context", {})
    

    dashboard_schema = json.dumps(context_data, indent=2)

    database_schema = """
Table Name: df

Columns:

- TransactionNumber : Transaction identifier
- RawCostInProjectCurrency : Actual transaction cost
- Quantity : Transaction quantity
- TaskId : Task identifier
- ProjectNumber : Project number
- ProjectName : Project name
- TaskNumber : Task number
- TaskName : Task name
- ExpenditureCategory : Expenditure category
- ExpenditureType : Expenditure type
- AccountingPeriod : Accounting period
- ProjectId : Project identifier
- Budget_RawCostAmounts : Budgeted cost
- Budget_RevenueAmounts : Budgeted revenue
- Cost_RawCostInProjectCurrency : Comparison cost
- Variance : Budget vs actual variance
"""

    template = """
You are an intelligent routing agent.

Your job is to decide which data source is required to answer the question.

Available Sources:

==================================================
SCREEN_ROUTE
==================================================

Use ONLY dashboard data.

Dashboard Data:

{dashboard_schema}


Examples:
- What is the CPI?
- What is the SPI?
- What is the project budget?
- What is the earned value?
- Show project health.
- Show project KPIs.
- What is the project manager name?

==================================================
DATABASE_ROUTE
==================================================

Use ONLY transaction table data.

Database Schema:

{database_schema}

Examples:
- Top expenditure types by cost.
- Highest cost tasks.
- Revenue by project.
- Variance by task.
- Show transactions.
- Show actual cost by category.
- Which task has highest spending?

==================================================
HYBRID_ROUTE
==================================================

Requires BOTH dashboard data and database data.

Use HYBRID_ROUTE when:

- Dashboard KPI needs transaction explanation.
- Dashboard metric needs drill-down.
- Question references both project KPIs and costs.
- User asks WHY a KPI changed.

Examples:

- Why is CPI low?
- Explain negative profit variance.
- Compare dashboard cost with actual transaction cost.
- Analyze project performance.
- Explain project health using expenditure data.
- Why is project behind schedule?
- What is causing the cost overrun shown on dashboard?

==================================================

Decision Rules:

1. If answer can be obtained only from dashboard fields:
   -> SCREEN_ROUTE

2. If answer can be obtained only from database table:
   -> DATABASE_ROUTE

3. If answer requires dashboard metrics AND transaction analysis:
   -> HYBRID_ROUTE

Question:
{question}

Return ONLY JSON:


  "route":"SCREEN_ROUTE"


or


  "route":"DATABASE_ROUTE"


or


  "route":"HYBRID_ROUTE"

"""

    prompt = PromptTemplate(
        input_variables=[
            "question",
            "dashboard_schema",
            "database_schema"
        ],
        template=template
    )

    structured_llm = llm_intent.with_structured_output(
        RouteDecision
    )

    response = structured_llm.invoke(
    prompt.format(
        question=state["question"],
        dashboard_schema=dashboard_schema,
        database_schema=database_schema
    ))

    #print("ROUTER RESPONSE =", response)
    #print("ROUTER RESPONSE TYPE =", type(response))

    if response is None:
        return {
            "route": "DATABASE_ROUTE"
        }

    return {
        "route": response.route
    }

def generate_query(state: QueryState):
    template = """
        You are an expert SQLite SQL generator for pandas DataFrames using sqldf.
        
        Task:
        Convert the user request into one valid SQLite SELECT query.
        
        Execution Context:
        - Library: sqldf
        - SQL Dialect: SQLite
        - Table Name: df
        - Number of Tables: 1
        - DataFrame is having 16 columns
        
        Table Description:
        The df table contains project transaction-level cost, budget, revenue, task, expenditure, accounting period, and variance data.
        
        Schema:
        - TransactionNumber TEXT
        - RawCostInProjectCurrency REAL
        - Quantity INTEGER
        - TaskId TEXT
        - ProjectNumber TEXT
        - ProjectName TEXT
        - TaskNumber TEXT
        - TaskName TEXT
        - ExpenditureCategory TEXT
        - ExpenditureType TEXT
        - AccountingPeriod TEXT
        - ProjectId TEXT
        - Budget_RawCostAmounts INTEGER
        - Budget_RevenueAmounts INTEGER
        - Cost_RawCostInProjectCurrency REAL
        - Variance REAL
        
        Column Meaning:
        - RawCostInProjectCurrency: actual transaction cost
        - Budget_RawCostAmounts: budgeted cost
        - Budget_RevenueAmounts: budgeted revenue
        - Cost_RawCostInProjectCurrency: comparison cost
        - Variance: difference between budget and cost
        - Quantity: transaction quantity
        - AccountingPeriod: accounting period as text
        - TaskName/TaskNumber/TaskId: task details
        - ProjectName/ProjectNumber/ProjectId: project details
        - ExpenditureCategory/ExpenditureType: expenditure classification
        - TransactionNumber: transaction identifier
        
        Important Rules:
        - Return ONLY SQL.
        - Do not explain the query.
        - Do not wrap SQL in markdown.
        - Use only SQLite-compatible syntax.
        - Use only DQL, mainly SELECT queries.
        - Do NOT use SELECT *.
        - Do NOT use JOIN because only one table exists.
        - Do NOT use DDL commands: CREATE, ALTER, DROP, TRUNCATE.
        - Do NOT use DML commands: INSERT, UPDATE, DELETE.
        - Do NOT use TCL commands: COMMIT, ROLLBACK, SAVEPOINT.
        - Do NOT use PRAGMA commands.
        - Do NOT invent columns.
        - Use only the given table name: df.
        - Use only the given column names.
        - If the request cannot be answered from the schema, return: CANNOT_ANSWER_FROM_SCHEMA.
        - For detail/list queries, always use LIMIT.
        - Default LIMIT is 50 if the user does not specify a limit.
        - For aggregation-only queries like SUM, AVG, COUNT, MIN, MAX, LIMIT is not required.
        - For GROUP BY queries, use LIMIT 50 unless the user asks for all results.
        
        
        Intent and Column Mapping Rules:
        - If user says "cost", "actual cost", "spent", or "expense", use RawCostInProjectCurrency.
        - If user says "budget cost", "budgeted cost", or "planned cost", use Budget_RawCostAmounts.
        - If user says "revenue", "budget revenue", or "planned revenue", use Budget_RevenueAmounts.
        - If user says "comparison cost", use Cost_RawCostInProjectCurrency.
        - If user says "variance", "difference", or "gap", use Variance.
        - If user says "task", use TaskName unless they specifically ask for TaskNumber or TaskId.
        - If user says "project", use ProjectName unless they specifically ask for ProjectNumber or ProjectId.
        - If user says "category", use ExpenditureCategory.
        - If user says "type", use ExpenditureType.
        - If user mentions a wrong or partial column name, infer the closest valid column from intent.
        - Always use exact column names from the schema.
        
        
        
        Ambiguity Handling:
        - If the user request is ambiguous but can be reasonably mapped to a column, generate the best matching query.
        - If multiple interpretations are equally possible, return CANNOT_ANSWER_FROM_SCHEMA.
        
        SQLite Compatibility:
        - Do not use ILIKE, TOP, DATE_TRUNC, EXTRACT, QUALIFY, or window functions unless necessary.
        - Use LIMIT instead of TOP.
        - Use LOWER(column) LIKE LOWER('%value%') for case-insensitive text search.
        
        
        Before returning SQL, silently verify:
        - The query uses only table df.
        - The query uses only schema columns.
        - The query is valid SQLite syntax.
        - The query is a SELECT query only.
        - No SELECT * is used.
        
        Return CANNOT_ANSWER_FROM_SCHEMA when:
        - The request needs external data.
        - The request asks for prediction, forecasting, or explanation not possible from columns.
        - The request asks to modify data.
        - The request asks for a column or metric that cannot be mapped to the schema.
        
        
        Examples:
        
        User: total cost
        SQL:
        SELECT SUM(RawCostInProjectCurrency) AS total_cost FROM df;
        
        User: cost by type
        SQL:
        SELECT ExpenditureType, SUM(RawCostInProjectCurrency) AS total_cost
        FROM df
        GROUP BY ExpenditureType
        ORDER BY total_cost DESC
        LIMIT 50;
        
        User: top 3 costly tasks
        SQL:
        SELECT TaskName, SUM(RawCostInProjectCurrency) AS total_cost
        FROM df
        GROUP BY TaskName
        ORDER BY total_cost DESC
        LIMIT 3;
        
        User: total budget cost
        SQL:
        SELECT SUM(Budget_RawCostAmounts) AS total_budget_cost FROM df;
        
        User: variance by task
        SQL:
        SELECT TaskName, SUM(Variance) AS total_variance
        FROM df
        GROUP BY TaskName
        ORDER BY total_variance DESC
        LIMIT 50;
        
        User: revenue by project
        SQL:
        SELECT ProjectName, SUM(Budget_RevenueAmounts) AS total_revenue
        FROM df
        GROUP BY ProjectName
        ORDER BY total_revenue DESC
        LIMIT 50;
        
        User: show transactions for labor
        SQL:
        SELECT TransactionNumber, ExpenditureType, RawCostInProjectCurrency, AccountingPeriod
        FROM df
        WHERE LOWER(ExpenditureType) LIKE LOWER('%labor%')
        LIMIT 50;
        
        User Request:
        {question}
        """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )

    response = query_generator.invoke(
        prompt.format(question=state["question"])
    )

    sql = clean_sql(response.content)

    return {
        "query": sql,
        "query_history": [sql]
    }

def evaluate_query(state: QueryState):
    template = """
        You are a strict SQLite SQL evaluator for pandas DataFrames using sqldf.
        
        Task:
        Evaluate whether the generated SQL correctly answers the user question using only the given schema and rules.
        
        Execution Context:
        - Library: sqldf
        - SQL Dialect: SQLite
        - Table Name: df
        - Number of Tables: 1
        - DataFrame has 16 columns
        
        Schema:
        - TransactionNumber TEXT
        - RawCostInProjectCurrency REAL
        - Quantity INTEGER
        - TaskId TEXT
        - ProjectNumber TEXT
        - ProjectName TEXT
        - TaskNumber TEXT
        - TaskName TEXT
        - ExpenditureCategory TEXT
        - ExpenditureType TEXT
        - AccountingPeriod TEXT
        - ProjectId TEXT
        - Budget_RawCostAmounts INTEGER
        - Budget_RevenueAmounts INTEGER
        - Cost_RawCostInProjectCurrency REAL
        - Variance REAL
        
        Column Meaning:
        - RawCostInProjectCurrency: actual transaction cost
        - Budget_RawCostAmounts: budgeted cost
        - Budget_RevenueAmounts: budgeted revenue
        - Cost_RawCostInProjectCurrency: comparison cost
        - Variance: difference between budget and cost
        - Quantity: transaction quantity
        - AccountingPeriod: accounting period as text
        - TaskName/TaskNumber/TaskId: task details
        - ProjectName/ProjectNumber/ProjectId: project details
        - ExpenditureCategory/ExpenditureType: expenditure classification
        - TransactionNumber: transaction identifier
        
        Evaluation Rules:
        - SQL must be valid SQLite syntax.
        - SQL must be a SELECT query only.
        - SQL must use only table name df.
        - SQL must use only columns listed in the schema.
        - SQL must not use SELECT *.
        - SQL must not use JOIN.
        - SQL must not use DDL, DML, TCL, or PRAGMA commands.
        - SQL must correctly match the intent of the user question.
        - SQL must use the correct metric column based on the user intent.
        - SQL must include LIMIT for detail/listing queries unless the user specifies another limit.
        - Aggregation-only queries such as SUM, AVG, COUNT, MIN, and MAX do not require LIMIT.
        - GROUP BY queries should include LIMIT unless the user asks for all results.
        - Case-insensitive text search should use LOWER(column) LIKE LOWER('%value%').
        - If the user question cannot be answered from the schema, the SQL should be CANNOT_ANSWER_FROM_SCHEMA.
        
        Intent Mapping:
        - "cost", "actual cost", "spent", or "expense" should map to RawCostInProjectCurrency.
        - "budget cost", "budgeted cost", or "planned cost" should map to Budget_RawCostAmounts.
        - "revenue", "budget revenue", or "planned revenue" should map to Budget_RevenueAmounts.
        - "comparison cost" should map to Cost_RawCostInProjectCurrency.
        - "variance", "difference", or "gap" should map to Variance.
        - "task" should map to TaskName unless TaskNumber or TaskId is specifically requested.
        - "project" should map to ProjectName unless ProjectNumber or ProjectId is specifically requested.
        - "category" should map to ExpenditureCategory.
        - "type" should map to ExpenditureType.
        
        Reject SQL when:
        - It uses wrong table names.
        - It uses columns not in the schema.
        - It uses SELECT *.
        - It uses JOIN.
        - It uses non-SQLite syntax such as ILIKE, TOP, DATE_TRUNC, EXTRACT, or QUALIFY.
        - It modifies data.
        - It answers a different question.
        - It misses required grouping, filtering, ordering, or limit.
        - It returns CANNOT_ANSWER_FROM_SCHEMA even though the question is answerable.
        - It generates SQL even though the question is not answerable from the schema.
        
        User Question:
        {question}
        
        Generated SQL:
        {query}
        
        Return ONLY valid JSON in this exact format:
        {{
          "evaluation": "approved" or "needs_improvement",
          "feedback": "short feedback"
        }}
        """

    prompt = PromptTemplate(
        input_variables=["question", "query"],
        template=template
    )

    response = query_evaluator.invoke(
        prompt.format(
            question=state["question"],
            query=state["query"]
        )
    )

    data = extract_json(response.content)

    evaluation = data.get("evaluation", "needs_improvement")
    feedback = data.get("feedback", "No feedback provided")

    if evaluation not in ["approved", "needs_improvement"]:
        evaluation = "needs_improvement"

    return {
        "evaluation": evaluation,
        "feedback": feedback,
        "feedback_history": [feedback]
    }

def optimize_query(state: QueryState):
    template = """
        You are an expert SQLite SQL optimizer and fixer for pandas DataFrames using sqldf.
        
        Task:
        Improve or correct the current SQL based on the user question and evaluator feedback.
        
        Execution Context:
        - Library: sqldf
        - SQL Dialect: SQLite
        - Table Name: df
        - Number of Tables: 1
        - DataFrame has 16 columns
        
        Schema:
        - TransactionNumber TEXT
        - RawCostInProjectCurrency REAL
        - Quantity INTEGER
        - TaskId TEXT
        - ProjectNumber TEXT
        - ProjectName TEXT
        - TaskNumber TEXT
        - TaskName TEXT
        - ExpenditureCategory TEXT
        - ExpenditureType TEXT
        - AccountingPeriod TEXT
        - ProjectId TEXT
        - Budget_RawCostAmounts INTEGER
        - Budget_RevenueAmounts INTEGER
        - Cost_RawCostInProjectCurrency REAL
        - Variance REAL
        
        Column Meaning:
        - RawCostInProjectCurrency: actual transaction cost
        - Budget_RawCostAmounts: budgeted cost
        - Budget_RevenueAmounts: budgeted revenue
        - Cost_RawCostInProjectCurrency: comparison cost
        - Variance: difference between budget and cost
        - Quantity: transaction quantity
        - AccountingPeriod: accounting period as text
        - TaskName/TaskNumber/TaskId: task details
        - ProjectName/ProjectNumber/ProjectId: project details
        - ExpenditureCategory/ExpenditureType: expenditure classification
        - TransactionNumber: transaction identifier
        
        Rules:
        - Return ONLY SQL.
        - Do not explain the query.
        - Do not wrap SQL in markdown.
        - Generate only SQLite-compatible SELECT queries.
        - Do NOT use SELECT *.
        - Do NOT use JOIN because only one table exists.
        - Do NOT use DDL commands: CREATE, ALTER, DROP, TRUNCATE.
        - Do NOT use DML commands: INSERT, UPDATE, DELETE.
        - Do NOT use TCL commands: COMMIT, ROLLBACK, SAVEPOINT.
        - Do NOT use PRAGMA commands.
        - Do NOT use window functions.
        - Do NOT invent columns.
        - Use only table name df.
        - Use only columns listed in the schema.
        - If the request cannot be answered from the schema, return: CANNOT_ANSWER_FROM_SCHEMA.
        - For detail/list queries, always use LIMIT.
        - Default LIMIT is 50 if the user does not specify a limit.
        - For aggregation-only queries like SUM, AVG, COUNT, MIN, MAX, LIMIT is not required.
        - For GROUP BY queries, use LIMIT 50 unless the user asks for all results.
        - Use LOWER(column) LIKE LOWER('%value%') for case-insensitive text search.
        - Use LIMIT instead of TOP.
        - Do not use unsupported SQLite syntax like ILIKE, TOP, DATE_TRUNC, EXTRACT, or QUALIFY.
        
        Intent Mapping:
        - If user says "cost", "actual cost", "spent", or "expense", use RawCostInProjectCurrency.
        - If user says "budget cost", "budgeted cost", or "planned cost", use Budget_RawCostAmounts.
        - If user says "revenue", "budget revenue", or "planned revenue", use Budget_RevenueAmounts.
        - If user says "comparison cost", use Cost_RawCostInProjectCurrency.
        - If user says "variance", "difference", or "gap", use Variance.
        - If user says "task", use TaskName unless they specifically ask for TaskNumber or TaskId.
        - If user says "project", use ProjectName unless they specifically ask for ProjectNumber or ProjectId.
        - If user says "category", use ExpenditureCategory.
        - If user says "type", use ExpenditureType.
        - If user mentions a wrong or partial column name, infer the closest valid column from intent.
        - Always use exact column names from the schema.
        
        Optimization Guidelines:
        - Fix syntax errors.
        - Fix wrong column names using the closest valid schema column.
        - Fix wrong metric selection based on user intent.
        - Add missing GROUP BY when aggregation is requested by category, task, project, type, or period.
        - Add missing ORDER BY for top, highest, lowest, or ranking questions.
        - Add missing LIMIT for detail/list or GROUP BY queries.
        - Remove unsupported SQLite syntax.
        - Remove unsafe or disallowed SQL commands.
        - Keep the query as simple as possible.
        - Do not change correct logic unnecessarily.
        
        User Question:
        {question}
        
        Current SQL:
        {query}
        
        Evaluator Feedback:
        {feedback}
        
        Return corrected SQL only.
        """

    prompt = PromptTemplate(
        input_variables=["question", "query", "feedback"],
        template=template
    )

    response = query_optimizer.invoke(
        prompt.format(
            question=state["question"],
            query=state["query"],
            feedback=state["feedback"]
        )
    )

    optimized_sql = clean_sql(response.content)

    return {
        "query": optimized_sql,
        "iteration": state["iteration"] + 1,
        "query_history": [optimized_sql]
    }


def result_summary(state: QueryState):
    summary_template = """
You are an expert data analyst.

Task:
Summarize the SQL query result in simple business language.

Context:
- The data comes from a pandas DataFrame queried using sqldf.
- The dataset contains project transaction-level cost, budget, revenue, task, expenditure, accounting period, and variance data.
- The SQL result is already computed.
- Do not create new numbers that are not present in the result.
- Do not mention SQL unless needed.
- Do not explain how the query was generated.

User Question:
{question}

SQL Query:
{query}

Query Result:
{result}

Instructions:
- Give a concise answer.
- Start with the direct answer to the user question.
- Mention the key values from the result.
- If the result has multiple rows, summarize the top findings.
- If the result is empty, say no matching records were found.
- If the result has only one value, explain what that value means.
- Use column names from the result, but convert them into readable business terms.
- Do not over-explain.
- Do not add assumptions.
- Do not provide Python code.
- Do not provide SQL code unless the user asks.

Return Format:
Answer:
<short natural language summary>
"""

    prompt = PromptTemplate(
        input_variables=["question", "query", "result"],
        template=summary_template
    )

    response = result_summarizer_llm.invoke(
        prompt.format(
            question=state["question"],
            query=state["query"],
            result=state["sql_result"]
        )
    )

    result_summary = response.content

    return {
        "result_summary":result_summary
    }


def screen_answer(state: QueryState):

    template = """
You are a project dashboard analyst.

Dashboard Data:

{dashboard}

Question:
{question}

Answer using ONLY dashboard data.
"""

    response = result_summarizer_llm.invoke(
        template.format(
            dashboard=json.dumps(
                state["dashboard_context"],
                indent=2
            ),
            question=state["question"]
        )
    )

    return {
        "result_summary": response.content
    }


# hybride

def generate_query_hybrid(state: QueryState):
    template = """
       You are an expert SQLite SQL Generator.

Your job is NOT to decide what to investigate.

The Investigation Planner has already analyzed the user request and produced an investigation plan.

You must generate SQL only for the database portion of the investigation.

--------------------------------------------------
USER QUESTION
--------------------------------------------------
{question}

--------------------------------------------------
INVESTIGATION PLAN
--------------------------------------------------
{investigation_plan}

--------------------------------------------------
DASHBOARD CONTEXT
--------------------------------------------------
{dashboard_context}

--------------------------------------------------
DATABASE SCHEMA
--------------------------------------------------

Table Name: df

Columns:

TransactionNumber TEXT
RawCostInProjectCurrency REAL
Quantity INTEGER
TaskId TEXT
ProjectNumber TEXT
ProjectName TEXT
TaskNumber TEXT
TaskName TEXT
ExpenditureCategory TEXT
ExpenditureType TEXT
AccountingPeriod TEXT
ProjectId TEXT
Budget_RawCostAmounts REAL
Budget_RevenueAmounts REAL
Cost_RawCostInProjectCurrency REAL
Variance REAL


Column Meaning:
- RawCostInProjectCurrency: actual transaction cost
- Budget_RawCostAmounts: budgeted cost
- Budget_RevenueAmounts: budgeted revenue
- Cost_RawCostInProjectCurrency: comparison cost
- Variance: difference between budget and cost
- Quantity: transaction quantity
- AccountingPeriod: accounting period as text
- TaskName/TaskNumber/TaskId: task details
- ProjectName/ProjectNumber/ProjectId: project details
- ExpenditureCategory/ExpenditureType: expenditure classification
- TransactionNumber: transaction identifier


--------------------------------------------------
OBJECTIVE
--------------------------------------------------

Generate SQL that retrieves ONLY the database information required by the Investigation Plan.

Do not answer the question.

Do not summarize.

Do not interpret dashboard values.

Only retrieve the data needed for later analysis.

--------------------------------------------------
RULES
--------------------------------------------------

- Return SQL only.
- SQLite syntax only.
- SELECT only.
- No DDL.
- No DML.
- No TCL.
- No JOIN.
- No SELECT *.
- Use only df.
- Use only schema columns.
- Use aliases for calculated fields.
- Use LIMIT when appropriate.
- Use CTEs when needed.
- Use aggregation when required.
- Return CANNOT_ANSWER_FROM_SCHEMA if database data cannot help.

--------------------------------------------------
HYBRID QUESTION RULE
--------------------------------------------------

For hybrid questions:

Dashboard values are already known.

Do NOT calculate dashboard metrics again.

Generate SQL only for supporting investigation data.

Dashboard metrics are already provided.

Never calculate:

- CPI
- SPI
- CV
- SV
- EAC
- Planned %
- Actual %
- Profit Variance

Only retrieve supporting transaction evidence.

Example:

Question:
"Why is CPI 0.93?"

Dashboard already provides CPI.

Do NOT calculate CPI.

Instead retrieve:

- highest cost categories
- largest variances
- cost overruns
- underperforming tasks

that help explain the CPI.

--------------------------------------------------
EXAMPLES
--------------------------------------------------

Question:
Why is CPI low?

Plan:
Investigate cost categories causing overspend.

SQL:

SELECT
    ExpenditureCategory,
    SUM(RawCostInProjectCurrency) AS total_actual_cost,
    SUM(Budget_RawCostAmounts) AS total_budget_cost,
    SUM(Variance) AS total_variance
FROM df
GROUP BY ExpenditureCategory
ORDER BY total_variance ASC
LIMIT 10;

Return EXACTLY ONE SQL statement.

Never generate multiple SELECT statements.

If multiple investigations are needed:

Choose the single most important investigation
that best answers the user's question.
        """

    prompt = PromptTemplate(
        input_variables=["question","investigation_plan","dashboard_context"],
        template=template
    )

    response = query_generator.invoke(
        prompt.format(question=state["question"],
                      investigation_plan = state["investigation_plan"],
                      dashboard_context = state["dashboard_context"])
    )

    sql = clean_sql(response.content)

    return {
        "query": sql,
        "query_history": [sql]
    }


def evaluate_query_hybrid(state: QueryState):
    template = """
       You are a Senior SQL Review Agent.

Your job is to validate whether the generated SQL correctly supports the Investigation Plan.

--------------------------------------------------
USER QUESTION
--------------------------------------------------
{question}

--------------------------------------------------
INVESTIGATION PLAN
--------------------------------------------------
{investigation_plan}

--------------------------------------------------
DASHBOARD CONTEXT
--------------------------------------------------
{dashboard_context}

--------------------------------------------------
GENERATED SQL
--------------------------------------------------
{query}

  
Schema:
- TransactionNumber TEXT
- RawCostInProjectCurrency REAL
- Quantity INTEGER
- TaskId TEXT
- ProjectNumber TEXT
- ProjectName TEXT
- TaskNumber TEXT
- TaskName TEXT
- ExpenditureCategory TEXT
- ExpenditureType TEXT
- AccountingPeriod TEXT
- ProjectId TEXT
- Budget_RawCostAmounts INTEGER
- Budget_RevenueAmounts INTEGER
- Cost_RawCostInProjectCurrency REAL
- Variance REAL

Column Meaning:
- RawCostInProjectCurrency: actual transaction cost
- Budget_RawCostAmounts: budgeted cost
- Budget_RevenueAmounts: budgeted revenue
- Cost_RawCostInProjectCurrency: comparison cost
- Variance: difference between budget and cost
- Quantity: transaction quantity
- AccountingPeriod: accounting period as text
- TaskName/TaskNumber/TaskId: task details
- ProjectName/ProjectNumber/ProjectId: project details
- ExpenditureCategory/ExpenditureType: expenditure classification
- TransactionNumber: transaction identifier

Rules:
- Return ONLY SQL.
- Do not explain the query.
- Do not wrap SQL in markdown.
- Generate only SQLite-compatible SELECT queries.
- Do NOT use SELECT *.
- Do NOT use JOIN because only one table exists.
- Do NOT use DDL commands: CREATE, ALTER, DROP, TRUNCATE.
- Do NOT use DML commands: INSERT, UPDATE, DELETE.
- Do NOT use TCL commands: COMMIT, ROLLBACK, SAVEPOINT.
- Do NOT use PRAGMA commands.
- Do NOT use window functions.
- Do NOT invent columns.
- Use only table name df.
- Use only columns listed in the schema.
- If the request cannot be answered from the schema, return: CANNOT_ANSWER_FROM_SCHEMA.
- For detail/list queries, always use LIMIT.
- Default LIMIT is 50 if the user does not specify a limit.
- For aggregation-only queries like SUM, AVG, COUNT, MIN, MAX, LIMIT is not required.
- For GROUP BY queries, use LIMIT 50 unless the user asks for all results.
- Use LOWER(column) LIKE LOWER('%value%') for case-insensitive text search.
- Use LIMIT instead of TOP.
- Do not use unsupported SQLite syntax like ILIKE, TOP, DATE_TRUNC, EXTRACT, or QUALIFY.
Reject SQL if it attempts to calculate dashboard metrics
already available in dashboard_context.

Intent Mapping:
- If user says "cost", "actual cost", "spent", or "expense", use RawCostInProjectCurrency.
- If user says "budget cost", "budgeted cost", or "planned cost", use Budget_RawCostAmounts.
- If user says "revenue", "budget revenue", or "planned revenue", use Budget_RevenueAmounts.
- If user says "comparison cost", use Cost_RawCostInProjectCurrency.
- If user says "variance", "difference", or "gap", use Variance.
- If user says "task", use TaskName unless they specifically ask for TaskNumber or TaskId.
- If user says "project", use ProjectName unless they specifically ask for ProjectNumber or ProjectId.
- If user says "category", use ExpenditureCategory.
- If user says "type", use ExpenditureType.
- If user mentions a wrong or partial column name, infer the closest valid column from intent.
- Always use exact column names from the schema.

Optimization Guidelines:
- Fix syntax errors.
- Fix wrong column names using the closest valid schema column.
- Fix wrong metric selection based on user intent.
- Add missing GROUP BY when aggregation is requested by category, task, project, type, or period.
- Add missing ORDER BY for top, highest, lowest, or ranking questions.
- Add missing LIMIT for detail/list or GROUP BY queries.
- Remove unsupported SQLite syntax.
- Remove unsafe or disallowed SQL commands.
- Keep the query as simple as possible.
- Do not change correct logic unnecessarily.

--------------------------------------------------
VALIDATION RULES
--------------------------------------------------

Check:

1. SQL syntax is valid SQLite.

2. Uses only df.

3. Uses only schema columns.

4. No SELECT *.

5. No JOIN.

6. No DDL/DML/TCL.

7. SQL supports Investigation Plan.

8. SQL retrieves the evidence required by the plan.

9. SQL does NOT try to recalculate dashboard metrics.

10. SQL retrieves explanatory data instead of dashboard values.

--------------------------------------------------
HYBRID VALIDATION
--------------------------------------------------

Question:
"Why is CPI 0.93?"

BAD SQL:

Calculate CPI.

GOOD SQL:

Retrieve:

- cost overruns
- high variance tasks
- expensive categories
- overspend drivers

Question:
"Why is EAC above budget?"

BAD SQL:

Recalculate EAC.

GOOD SQL:

Retrieve:

- largest budget overruns
- highest variance categories
- tasks with overspend

--------------------------------------------------
RETURN JSON ONLY

{{
  "evaluation":"approved",
  "feedback":""
}}

or

{{
  "evaluation":"needs_improvement",
  "feedback":"SQL is calculating CPI instead of retrieving root-cause data."
}}
        """

    prompt = PromptTemplate(
        input_variables=["question", "query","dashboard_context","investigation_plan"],
        template=template
    )

    response = query_evaluator.invoke(
        prompt.format(
            question=state["question"],
            query=state["query"],
            dashboard_context = state["dashboard_context"],
            investigation_plan = state["investigation_plan"]
        )
    )

    data = extract_json(response.content)

    evaluation = data.get("evaluation", "needs_improvement")
    feedback = data.get("feedback", "No feedback provided")

    if evaluation not in ["approved", "needs_improvement"]:
        evaluation = "needs_improvement"

    return {
        "evaluation": evaluation,
        "feedback": feedback,
        "feedback_history": [feedback]
    }

def optimize_query_hybrid(state: QueryState):
    template = """
       You are a SQL Optimization Agent.

Your job is to improve SQL using:

- User Question
- Investigation Plan
- Dashboard Context
- Evaluator Feedback

--------------------------------------------------
USER QUESTION
--------------------------------------------------
{question}

--------------------------------------------------
INVESTIGATION PLAN
--------------------------------------------------
{investigation_plan}

--------------------------------------------------
DASHBOARD CONTEXT
--------------------------------------------------
{dashboard_context}

--------------------------------------------------
CURRENT SQL
--------------------------------------------------
{query}

--------------------------------------------------
EVALUATOR FEEDBACK
--------------------------------------------------
{feedback}

--------------------------------------------------
DATABASE SCHEMA
--------------------------------------------------

Schema:
- TransactionNumber TEXT
- RawCostInProjectCurrency REAL
- Quantity INTEGER
- TaskId TEXT
- ProjectNumber TEXT
- ProjectName TEXT
- TaskNumber TEXT
- TaskName TEXT
- ExpenditureCategory TEXT
- ExpenditureType TEXT
- AccountingPeriod TEXT
- ProjectId TEXT
- Budget_RawCostAmounts INTEGER
- Budget_RevenueAmounts INTEGER
- Cost_RawCostInProjectCurrency REAL
- Variance REAL

Column Meaning:
- RawCostInProjectCurrency: actual transaction cost
- Budget_RawCostAmounts: budgeted cost
- Budget_RevenueAmounts: budgeted revenue
- Cost_RawCostInProjectCurrency: comparison cost
- Variance: difference between budget and cost
- Quantity: transaction quantity
- AccountingPeriod: accounting period as text
- TaskName/TaskNumber/TaskId: task details
- ProjectName/ProjectNumber/ProjectId: project details
- ExpenditureCategory/ExpenditureType: expenditure classification
- TransactionNumber: transaction identifier

Rules:
- Return ONLY SQL.
- Do not explain the query.
- Do not wrap SQL in markdown.
- Generate only SQLite-compatible SELECT queries.
- Do NOT use SELECT *.
- Do NOT use JOIN because only one table exists.
- Do NOT use DDL commands: CREATE, ALTER, DROP, TRUNCATE.
- Do NOT use DML commands: INSERT, UPDATE, DELETE.
- Do NOT use TCL commands: COMMIT, ROLLBACK, SAVEPOINT.
- Do NOT use PRAGMA commands.
- Do NOT use window functions.
- Do NOT invent columns.
- Use only table name df.
- Use only columns listed in the schema.
- If the request cannot be answered from the schema, return: CANNOT_ANSWER_FROM_SCHEMA.
- For detail/list queries, always use LIMIT.
- Default LIMIT is 50 if the user does not specify a limit.
- For aggregation-only queries like SUM, AVG, COUNT, MIN, MAX, LIMIT is not required.
- For GROUP BY queries, use LIMIT 50 unless the user asks for all results.
- Use LOWER(column) LIKE LOWER('%value%') for case-insensitive text search.
- Use LIMIT instead of TOP.
- Do not use unsupported SQLite syntax like ILIKE, TOP, DATE_TRUNC, EXTRACT, or QUALIFY.
If SQL calculates dashboard metrics,
remove those calculations and replace
them with supporting evidence queries.

Intent Mapping:
- If user says "cost", "actual cost", "spent", or "expense", use RawCostInProjectCurrency.
- If user says "budget cost", "budgeted cost", or "planned cost", use Budget_RawCostAmounts.
- If user says "revenue", "budget revenue", or "planned revenue", use Budget_RevenueAmounts.
- If user says "comparison cost", use Cost_RawCostInProjectCurrency.
- If user says "variance", "difference", or "gap", use Variance.
- If user says "task", use TaskName unless they specifically ask for TaskNumber or TaskId.
- If user says "project", use ProjectName unless they specifically ask for ProjectNumber or ProjectId.
- If user says "category", use ExpenditureCategory.
- If user says "type", use ExpenditureType.
- If user mentions a wrong or partial column name, infer the closest valid column from intent.
- Always use exact column names from the schema.

Optimization Guidelines:
- Fix syntax errors.
- Fix wrong column names using the closest valid schema column.
- Fix wrong metric selection based on user intent.
- Add missing GROUP BY when aggregation is requested by category, task, project, type, or period.
- Add missing ORDER BY for top, highest, lowest, or ranking questions.
- Add missing LIMIT for detail/list or GROUP BY queries.
- Remove unsupported SQLite syntax.
- Remove unsafe or disallowed SQL commands.
- Keep the query as simple as possible.
- Do not change correct logic unnecessarily.

--------------------------------------------------
OBJECTIVE
--------------------------------------------------

Modify SQL so that it fully supports the Investigation Plan.

Do not answer the question.

Do not summarize.

Do not calculate dashboard metrics.

Only retrieve supporting database evidence.

--------------------------------------------------
FIXES
--------------------------------------------------

Correct:

- wrong columns
- wrong filters
- missing aggregations
- missing ordering
- missing limits
- invalid syntax
- unnecessary calculations

--------------------------------------------------
HYBRID RULE
--------------------------------------------------

Dashboard metrics already exist.

Never regenerate:

- CPI
- SPI
- EAC
- CV
- SV
- Profit Variance
- Planned %
- Actual %

Instead retrieve evidence that explains them.

Examples:

If dashboard says:

CPI = 0.93

Retrieve:

- overspending categories
- negative variance tasks
- highest actual costs

If dashboard says:

EAC above budget

Retrieve:

- biggest budget overruns
- variance contributors
- costly expenditure types

--------------------------------------------------
RETURN
--------------------------------------------------

SQL only.
        """

    prompt = PromptTemplate(
        input_variables=["question", "query", "feedback","dashboard_context","investigation_plan"],
        template=template
    )

    response = query_optimizer.invoke(
        prompt.format(
            question=state["question"],
            query=state["query"],
            feedback=state["feedback"],
            investigation_plan = state["investigation_plan"],
            dashboard_context = state["dashboard_context"]
        )
    )

    optimized_sql = clean_sql(response.content)

    return {
        "query": optimized_sql,
        "iteration": state["iteration"] + 1,
        "query_history": [optimized_sql]
    }

def investigation_planner(state: QueryState):

    template = """
You are a Project Controls Investigation Planner.

Your job is NOT to generate SQL.

Your job is to analyze the user's question and create an investigation plan
that identifies:

1. What dashboard metrics are relevant.
2. What database evidence is required.
3. What SQL should retrieve.
4. What final answer should explain.

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
DASHBOARD CONTEXT
--------------------------------------------------

{dashboard_context}

--------------------------------------------------
AVAILABLE DATABASE COLUMNS
--------------------------------------------------
--------------------------------------------------
DATABASE SCHEMA
--------------------------------------------------

Schema:
- TransactionNumber TEXT
- RawCostInProjectCurrency REAL
- Quantity INTEGER
- TaskId TEXT
- ProjectNumber TEXT
- ProjectName TEXT
- TaskNumber TEXT
- TaskName TEXT
- ExpenditureCategory TEXT
- ExpenditureType TEXT
- AccountingPeriod TEXT
- ProjectId TEXT
- Budget_RawCostAmounts INTEGER
- Budget_RevenueAmounts INTEGER
- Cost_RawCostInProjectCurrency REAL
- Variance REAL

Column Meaning:
- RawCostInProjectCurrency: actual transaction cost
- Budget_RawCostAmounts: budgeted cost
- Budget_RevenueAmounts: budgeted revenue
- Cost_RawCostInProjectCurrency: comparison cost
- Variance: difference between budget and cost
- Quantity: transaction quantity
- AccountingPeriod: accounting period as text
- TaskName/TaskNumber/TaskId: task details
- ProjectName/ProjectNumber/ProjectId: project details
- ExpenditureCategory/ExpenditureType: expenditure classification
- TransactionNumber: transaction identifier

--------------------------------------------------
RULES
--------------------------------------------------

Determine:

A. Dashboard Metrics To Use

Examples:
- CPI
- SPI
- EAC
- CV
- SV
- Profit Variance
- Actual %
- Planned %
- Cost
- Budget

B. Database Evidence Required

Examples:
- highest cost tasks
- biggest negative variance tasks
- overspending expenditure types
- cost categories causing overruns
- accounting periods with largest variance

C. SQL Retrieval Goal

Examples:
- retrieve top overspending tasks
- retrieve highest variance categories
- retrieve largest cost contributors

D. Final Explanation Goal

Examples:
- explain low CPI
- explain schedule slippage
- explain profit variance

--------------------------------------------------
RETURN FORMAT
--------------------------------------------------

Return output in the form of string .

"

"dashboard_metrics":[...],
"database_evidence":[...],
"sql_goal":"...",
"answer_goal":"..."
 "
 
"""

    prompt = PromptTemplate(
        input_variables=["question", "dashboard_context"],
        template=template
    )

    response = query_generator.invoke(
        prompt.format(
            question=state["question"],
            dashboard_context=state["dashboard_context"]
        )
    )

    return {
        "investigation_plan": response.content
    }



def screen_answer_node(state: QueryState):

    template = """
You are a Project Cost Control Dashboard Assistant.

Answer the user's question using ONLY the dashboard context.

--------------------------------------------------
USER QUESTION
--------------------------------------------------
{question}

--------------------------------------------------
DASHBOARD CONTEXT
--------------------------------------------------
{dashboard_context}

--------------------------------------------------
RULES
--------------------------------------------------

- Use ONLY dashboard data.
- Do NOT generate SQL.
- Do NOT assume values that do not exist.
- If the answer is available directly from dashboard data,
  answer it clearly.
- If the answer cannot be determined from dashboard data,
  explain what information is missing.
- Convert technical metrics into business language.
- Be concise.
- Do not mention JSON.
- Do not mention database tables.
- Do not mention SQL.

Examples:

Question:
What is the CPI?

Answer:
The Cost Performance Index (CPI) is 0.93, indicating the project is spending slightly more than the value earned.

Question:
What is the project budget?

Answer:
The approved project budget is AED 162.1M.

Question:
What is the SPI?

Answer:
The Schedule Performance Index (SPI) is 0.96, indicating the project is progressing slightly behind plan.

Question:
Can you confirm if approved change orders are included in the budget baseline?

Answer:
The dashboard does not contain information about approved change orders, so this cannot be confirmed from the available data.

--------------------------------------------------
RETURN
--------------------------------------------------

Return only the answer.
"""

    prompt = PromptTemplate(
        input_variables=[
            "question",
            "dashboard_context"
        ],
        template=template
    )

    response = result_summarizer_llm.invoke(
        prompt.format(
            question=state["question"],
            dashboard_context=json.dumps(
                state["dashboard_context"],
                indent=2
            )
        )
    )

    return {
        "result_summary": response.content
    }


def run_sql_node(state: QueryState):
    sql_query = state["query"]

    if "CANNOT_ANSWER_FROM_SCHEMA" in sql_query:
        return {
            "sql_result": "Query cannot be answered from schema."
        }

    try:
        result = sqldf(sql_query, {"df": df})
        return {
            "sql_result": result.to_string(index=False)
        }

    except Exception as e:
        return {
            "sql_result": f"SQL Error: {e}"
        }


builder = StateGraph(QueryState)

builder.add_node(
    "intent_router",
    intent_router
)


builder.add_node(
    "screen_answer",
    screen_answer_node
)


builder.add_node(
    "investigation_planner",
    investigation_planner
)

builder.add_node(
    "generate_query_hybrid",
    generate_query_hybrid
)

builder.add_node(
    "evaluate_query_hybrid",
    evaluate_query_hybrid
)

builder.add_node(
    "optimize_query_hybrid",
    optimize_query_hybrid
)

builder.add_node(
    "generate_query",
    generate_query
)

builder.add_node(
    "evaluate_query",
    evaluate_query
)

builder.add_node(
    "optimize_query",
    optimize_query
)

builder.add_node(
    "run_sql",
    run_sql_node
)

builder.add_node(
    "result_summary",
    result_summary
)


def route_after_intent(state: QueryState):

    route = state["route"]

    if route == "SCREEN_ROUTE":
        return "screen_answer"

    elif route == "DATABASE_ROUTE":
        return "generate_query"

    else:
        return "investigation_planner"


def route_after_database_evaluation(state: QueryState):

    if state["evaluation"] == "approved":
        return "run_sql"

    if state["iteration"] >= state["max_iteration"]:
        return "run_sql"

    return "optimize_query"


def route_after_hybrid_evaluation(state: QueryState):

    if state["evaluation"] == "approved":
        return "run_sql"

    if state["iteration"] >= state["max_iteration"]:
        return "run_sql"

    return "optimize_query_hybrid"


builder.add_edge(
    START,
    "intent_router"
)

builder.add_conditional_edges(
    "intent_router",
    route_after_intent,
    {
        "screen_answer": "screen_answer",
        "generate_query": "generate_query",
        "investigation_planner": "investigation_planner"
    }
)

builder.add_edge(
    "screen_answer",
    END
)

builder.add_edge(
    "generate_query",
    "evaluate_query"
)

builder.add_conditional_edges(
    "evaluate_query",
    route_after_database_evaluation,
    {
        "optimize_query": "optimize_query",
        "run_sql": "run_sql"
    }
)

builder.add_edge(
    "optimize_query",
    "evaluate_query"
)

builder.add_edge(
    "investigation_planner",
    "generate_query_hybrid"
)

builder.add_edge(
    "generate_query_hybrid",
    "evaluate_query_hybrid"
)

builder.add_conditional_edges(
    "evaluate_query_hybrid",
    route_after_hybrid_evaluation,
    {
        "optimize_query_hybrid": "optimize_query_hybrid",
        "run_sql": "run_sql"
    }
)

builder.add_edge(
    "optimize_query_hybrid",
    "evaluate_query_hybrid"
)

builder.add_edge(
    "run_sql",
    "result_summary"
)

builder.add_edge(
    "result_summary",
    END
)

sql_app = builder.compile()

def executeSql(payload: dict):
    initial_state = {
        "question": payload["question"],
        "route": "",
        "dashboard_context": payload["dashboard_context"],
        "investigation_plan": "",
        "query": "",
        "evaluation": "needs_improvement",
        "feedback": "",
        "iteration": 0,
        "max_iteration": 5,
        "sql_result": "",
        "result_summary": "",
        "query_history": [],
        "feedback_history": []
    }


    final_state = sql_app.invoke(
        initial_state
    )
    return final_state

# print(executeSql({
#     "question":"Get top 5 records by cost",
#     "dashboard_context":{
#   "project": {
#     "actualPercent": "50.70",
#     "EstimatedAmount": "171800000",
#     "budget": "72.45M",
#     "cost": "36.73M",
#     "CPI": "1.00",
#     "CV": "0",
#     "EAC": "79.82M",
#     "earnedValue": "36.73M",
#     "plannedPercent": "56.3",
#     "plannedValue": "91200000",
#     "profitVariance": "-9800000",
#     "projectName": "112-ASD2",
#     "ProjectId": "300000004277391",
#     "ProjectManagerName": "Integration User",
#     "ProjectNumber": "112-P1131",
#     "ProjectStartDate": "2023-06-07",
#     "SPI": "1.25",
#     "SV": "7.37M"
#   },
#   "periodData": [
#     {
#       "period": "Jan",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Feb",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Mar",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Apr",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "May",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Jun",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Jul",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Aug",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Sep",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Oct",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Nov",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Dec",
#       "type": "budget",
#       "cost": 797866.7400881
#     },
#     {
#       "period": "Jan",
#       "type": "cost",
#       "cost": -366721.62
#     },
#     {
#       "period": "Feb",
#       "type": "cost",
#       "cost": 14873.03
#     },
#     {
#       "period": "Mar",
#       "type": "cost",
#       "cost": 160234.75
#     },
#     {
#       "period": "Apr",
#       "type": "cost",
#       "cost": 16958.45
#     },
#     {
#       "period": "May",
#       "type": "cost",
#       "cost": 174655.44
#     },
#     {
#       "period": "Jun",
#       "type": "cost",
#       "cost": 0
#     },
#     {
#       "period": "Jul",
#       "type": "cost",
#       "cost": 898030
#     },
#     {
#       "period": "Aug",
#       "type": "cost",
#       "cost": 63530.229999999996
#     },
#     {
#       "period": "Sep",
#       "type": "cost",
#       "cost": -16580.86
#     },
#     {
#       "period": "Oct",
#       "type": "cost",
#       "cost": 0
#     },
#     {
#       "period": "Nov",
#       "type": "cost",
#       "cost": 0
#     },
#     {
#       "period": "Dec",
#       "type": "cost",
#       "cost": 24300
#     }
#   ],
#   "categoryData": [
#     {
#       "Budget": "0",
#       "Cost": "2.25",
#       "Percentage": 0,
#       "ResourceClassName": "Labor"
#     },
#     {
#       "Budget": "0",
#       "Cost": "2.03M",
#       "Percentage": 0,
#       "ResourceClassName": "Equipment"
#     },
#     {
#       "Budget": "0",
#       "Cost": "6.11M",
#       "Percentage": 0,
#       "ResourceClassName": "Material Items"
#     },
#     {
#       "Budget": "3.13M",
#       "Cost": "15.74M",
#       "Percentage": 503.51,
#       "ResourceClassName": "Financial Resources"
#     }
#   ]
# }
# }))

# print("\nQuestion:")
# print(final_state["question"])

# print("\nRoute:")
# print(final_state.get("route"))

# print("\nInvestigation Plan:")
# print(final_state.get("investigation_plan"))

# print("\nSQL Query:")
# print(final_state.get("query"))

# print("\nSQL Result:")
# print(final_state.get("sql_result"))

# print("\nFinal Summary:")
# print(final_state.get("result_summary"))