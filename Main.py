from src.required_classes import modelOperations,dataAnalytics
import matplotlib.pyplot as plt

a =modelOperations()
df= a.model_inference()

data = dataAnalytics(df)

data.by_category()

data.by_day()