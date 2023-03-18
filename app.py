import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pickle import load

from pydoc import locate
from typing import List
import uvicorn

class InputFeatures(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

MODEL_PATH = "./model/xgb_model.pkl"

def load_model(path):
    model = load(open(path, "rb"))
    return model

def create_type_instance(type_name: str):
    return locate(type_name).__call__()

def get_features_dict(model):
    feature_names = model.get_booster().feature_names
    feature_types = list(map(create_type_instance, model.get_booster().feature_types))
    return dict(zip(feature_names, feature_types))

def create_input_features_class(model):
    return type("InputFeatures", (BaseModel,), get_features_dict(model))

model = load_model(MODEL_PATH)
InputFeatures = create_input_features_class(model)
app = FastAPI()

@app.post("/predict", response_model=List)
async def predict_post(datas: List[InputFeatures]):
    return model.predict(np.asarray([list(data.__dict__.values()) for data in datas])).tolist()

if __name__ == "__main__":
    print(get_features_dict(model))
    uvicorn.run(app, host="0.0.0.0", port=8000)