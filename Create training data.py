import json
import openai
import pandas as pd

api_key ="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"
openai.api_key = api_key

# 讀取 Excel 檔案
df = pd.read_excel("data.xlsx", sheet_name="training_data")

# 將資料轉換為 JSON 格式，以 ID 為區分，加入標記
data = {}
for index, row in df.iterrows():
    id = row["ID"]
    prompt = row["Prompt"]
    completion = row["Completion"]
    tags = [tag.strip() for tag in row["Tags"].split(",") if tag.strip() in ["order", "product"]] if not pd.isna(row["Tags"]) else []
    if id not in data:
        data[id] = []
    data[id].append({"prompt": prompt, "completion": completion, "tags": tags})

# 將資料寫入到檔案中
with open("training_data.jsonl", "w") as outfile:
    for id, items in data.items():
        for item in items:
            json.dump(item, outfile)
            outfile.write('\n')

# 讀取訓練資料
training_data = []
with open("training_data.jsonl", "r") as infile:
    for line in infile:
        training_data.append(json.loads(line))

# 上傳訓練資料到 OpenAI
file_name = "training_data.jsonl"
upload_response = openai.File.create(file=open(file_name, "rb"), purpose="fine-tune")
file_id = upload_response.id
print("Training data uploaded. File ID:", file_id)

# 訓練模型
model_engine = "davinci" # 設定 GPT-3 引擎
fine_tune_response = openai.FineTune.create(training_file=file_id, model=model_engine)
model_id = fine_tune_response.model_id
print("Model fine-tuned. Model ID:", model_id)

# 定義回答函數
def get_model_response(prompt):
    response = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024)
    if len(response.choices) > 0:
        return response.choices[0].text.strip()
    else:
        return ""

# 判斷問題是否在標記範圍內，如果不在則回覆預設回答
def process_question(question):
    tags = ["order", "product", "support"]
    for tag in tags:
        if tag in question:
            # 在範圍內，回傳模型的回答
            return get_model_response(question)
    # 不在範圍內，回傳預設回答
    return "很抱歉，我不確定我是否可以回答您的問題，請您詢問相關問題。"

# 取得所有模型清單
models = openai.Model.list()

# 找到剛剛fine-tune的模型ID
model_id = fine_tune_response.model_id

# 取得fine-tune後的模型物件
fine_tuned_model = None
for model in models["data"]:
    if model["id"] == model_id:
        fine_tuned_model = openai.Model.retrieve(model_id)
        break

# 將fine-tune後的模型物件存入變數中
model_engine = fine_tuned_model["id"]
