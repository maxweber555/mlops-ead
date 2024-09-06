import os
import json
import mlflow
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI


class FetalHelthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


app = FastAPI(title="Fetal Health API",
              openapi_tags=[
                    {
                        "name": "Health",
                        "description": "Get api health"
                    },
                  {
                      "name": "Prediction",
                      "description": "Model prediction"
                    }
              ])


def load_model():
    print('reading model .....')
    MLFLOW_TRACKING_URI = 'https://dagshub.com/maxweber555/my-first-repo.mlflow'
    MLFLOW_TRACKING_USERNAME = 'maxweber555'
    MLFLOW_TRACKING_PASSWORD = 'ffcbe8103971572a59dd6dc89c89233d183dbf64'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    print('setting mlflow')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print('Create client.....')
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print('getting registered Model.....')
    registered_model = client.get_registered_model('fetal_healh')
    print('read model....')
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(loaded_model)
    return loaded_model


@app.on_event(event_type='startup')
def startup_even():
    global loaded_model
    loaded_model = load_model()
    pass

@app.get(path='/',
         tags=["Health"])
def api_health():
    return {"status": "helthy"}


@app.post(path='/predict',
         tags=['Prediction'])
def predict(request: FetalHelthData):
    global loaded_model
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)

    prediction=loaded_model.predict(received_data)
    print(received_data)
    print(prediction)
    return {"prediction": str(np.argmax(prediction[0]))}

