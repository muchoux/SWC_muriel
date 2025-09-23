from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend_step_by_step import run_simulation
from models import Results, CalculateInput

app = FastAPI()

ENV = os.getenv('ENV', default='dev')

if ENV == "prod":
    # Servir les fichiers statiques React
    app.mount("/static", StaticFiles(directory="static/build/static"), name="static")

    # Route catch-all pour React Router
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        index_path = os.path.join("static", "build", "index.html")
        return FileResponse(index_path)

@app.get("/tests")
def hello_world():
    return {"message : Hello World"}

@app.post("/calculate")
def calculate(input_data : CalculateInput ) -> Results:
    
    print(input_data)
    
    simulation_values = run_simulation(source = input_data.model , d = input_data.economic_parameters.d/100,
                            rom=input_data.economic_parameters.rOM/100,
                            rps=input_data.economic_parameters.rps/100,
                            rd=input_data.technical_parameters.rd/100, 
                            rpg=input_data.economic_parameters.rpg/100, 
                            P=input_data.technical_parameters.P,
                            Cu=input_data.economic_parameters.Cu,
                            Hopt=input_data.technical_parameters.Hopt, 
                            PR=input_data.technical_parameters.PR/100,
                            SCI=input_data.technical_parameters.SCI/100,
                            COM=input_data.economic_parameters.COM/100,
                            N=input_data.technical_parameters.N,
                            Xl=input_data.financial_parameters.Xl/100,
                            Xec=input_data.financial_parameters.Xec/100,
                            il=input_data.financial_parameters.il/100,
                            dec=input_data.financial_parameters.dec/100,
                            T=input_data.economic_parameters.T/100,
                            Xd=input_data.economic_parameters.Xd/100,
                            Nd=input_data.economic_parameters.Nd,
                            Xis=input_data.financial_parameters.Xis/100,
                            Nis=input_data.financial_parameters.Nis,
                            Nl=input_data.financial_parameters.Nl,
                            ps=input_data.economic_parameters.ps,
                            pg=input_data.economic_parameters.pg)
    
    msg = f"Your data has been processed"
    result = Results(Result=simulation_values, msg=msg)
    return result

app.add_middleware(CORSMiddleware, 
    allow_origins = ["*"], allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)