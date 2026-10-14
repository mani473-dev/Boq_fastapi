from main import (
    executeSql,
    get_all_records,
    create_clean_dataframe,

    SUPPLIER_RESPONSE_COLUMN_MAPPING,
    LINE_SUPPLIER_DETAILS_COLUMN_MAPPING,
    SECTION_SCORING_COLUMN_MAPPING,

    Section_Scoring_Details_url,
    Line_Supplier_Details_url,
    Supplier_Response_url,
    execution_times


)
import pandas as pd



def getOutput(question:str):

    supplier_response_df = create_clean_dataframe(
        get_all_records(Supplier_Response_url),SUPPLIER_RESPONSE_COLUMN_MAPPING)
    
    line_supplier_df = create_clean_dataframe(
        get_all_records(Line_Supplier_Details_url),LINE_SUPPLIER_DETAILS_COLUMN_MAPPING)
    


    section_scoring_df = create_clean_dataframe(
        get_all_records(Section_Scoring_Details_url),SECTION_SCORING_COLUMN_MAPPING
    )
    

    response = executeSql(question,supplier_response_df,line_supplier_df,section_scoring_df)
    print(execution_times)
    total_time = 0
    for _, t in execution_times.items():
      total_time += t
    print("sum of all time seconds ", total_time)

    return response



if __name__ == "__main__":
  

    response = getOutput("Key differences between Al Menha Tailoring and Embroidery LLC and STONE HERITAGE L.L.C.?")

    print(response.get("result_summary"))

    print("result table  is : ")

    print(response.get("sql_result"))

    print("sql query is  ")

    print(response.get("query"))
