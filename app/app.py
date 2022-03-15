import pandas as pd
from autogluon.tabular import TabularPredictor

model = TabularPredictor.load('/opt/ml/model')
model.persist_models(models='all')


# Lambda handler code
def lambda_handler(event, context):
    print(event['body'])
    df = pd.read_json(event['body'])
    print(df)
    pred_probs = model.predict_proba(df)

    return {
        'statusCode': 200,
        'body': pred_probs.to_json()
    }
